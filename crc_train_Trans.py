import warnings

import pandas as pd
from gentrl.utils import CharVocab

from dffndds.dataset import SynergyEncoderDataset
from dffndds.model_h import DualInteract

warnings.filterwarnings('ignore')
import os, random
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader
from tqdm import trange

from gentrl.encoder_Trans import TransEncoder
from chemprop.parsing import add_train_args, modify_train_args
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data
from chemprop.utils import get_loss_func, get_metric_func, makedirs, save_checkpoint

from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, recall_score, precision_score, \
    f1_score
from dffndds.entry import read_data_file, compute_kl_loss

torch.backends.cudnn.benchmark = False
device = torch.device("cuda")


dti_df = pd.read_csv("./data/crc/dti.csv")
VOCAB = CharVocab.from_data(dti_df['smiles'])

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps`
        steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(
            max(1.0, self.t_total - self.warmup_steps)))


class DiseaseModel(nn.Module):

    def __init__(self, args):
        super(DiseaseModel, self).__init__()
        proj_dim = 256
        dropout_rate = 0.5

        self.encoder = TransEncoder(VOCAB, bs=args.batch_size, latent_size=args.latent_size)

        self.num_crc_targets = args.num_crc_targets
        self.crc_ffn = nn.Linear(args.latent_size, args.num_tasks)
        self.ffn = [self.crc_ffn]

        ## comb net
        self.projection_context = nn.Sequential(
            # nn.LayerNorm(112),#改变norm的维度
            nn.Linear(18046, proj_dim),  # drugcombdb是112,drugbankddi是86,288
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_fp1 = nn.Sequential(
            # nn.LayerNorm(1024),
            nn.Linear(1024, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_fp2 = nn.Sequential(
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            nn.Linear(1024, proj_dim),
            nn.ReLU(),
            # nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.feature_interact = DualInteract(field_dim=5, embed_size=proj_dim, head_num=4)
        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim * 5),
            nn.Linear(proj_dim * 5, 2),
        )

        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)


    def DTI_forward(self, smiles_batch, mode):
        DTI_vecs = self.encoder.encode(smiles_batch)
        DTI_vecs = torch.sigmoid(DTI_vecs)
        crc_vecs = DTI_vecs[:, :self.num_crc_targets]
        normal_vecs = DTI_vecs[:, self.num_crc_targets:]
        crc_vecs = torch.cat([crc_vecs, normal_vecs], dim=-1)
        DTI_splitted = [crc_vecs]
        return DTI_splitted[mode]

    def forward(self, smiles_batch, mode):
        DTI_vecs = self.DTI_forward(smiles_batch, mode)
        return self.ffn[mode](DTI_vecs)

    def combo_forward(self, smiles1, smiles2, compound_1, compound_2, context, fp1_vectors, fp2_vectors, mode):
        DTI_vecs1 = self.DTI_forward(smiles1, mode)
        DTI_vecs2 = self.DTI_forward(smiles2, mode)
        contextFeatures = self.projection_context(context)
        fp1_vectors = self.projection_fp1(fp1_vectors)
        fp2_vectors = self.projection_fp2(fp2_vectors)
        all_features = torch.stack(
            [DTI_vecs1, DTI_vecs2, contextFeatures.squeeze(1), fp1_vectors, fp2_vectors],
            dim=1)
        all_features = self.feature_interact(all_features)
        out = self.transform(all_features)
        return out


def prepare_data(args):
    ## prepare dti data
    dti_data = get_data(path=args.dti_path, args=args)

    ## prepare single agent data
    args.use_compound_names = True
    single_data = get_data(path=args.single_path, args=args)
    args.use_compound_names = False

    ## prepare combine data
    ## train dataset
    smiles_1, smiles_2, context, Y = read_data_file('./data/crc/drugcombdb/synergy_train.csv')
    comb_train_dataset = SynergyEncoderDataset(np.array(smiles_1),
                                               np.array(smiles_2),
                                               np.array(Y),
                                               np.array(context),
                                               128, device=torch.device("cuda"))
    ## valid dataset
    smiles_1, smiles_2, context, Y = read_data_file('./data/crc/drugcombdb/synergy_valid.csv')
    comb_valid_dataset = SynergyEncoderDataset(np.array(smiles_1),
                                               np.array(smiles_2),
                                               np.array(Y),
                                               np.array(context),
                                               128, device=torch.device("cuda"))
    ## test dataset
    smiles_1, smiles_2, context, Y = read_data_file('./data/crc/drugcombdb/synergy_test.csv')
    comb_test_dataset = SynergyEncoderDataset(np.array(smiles_1),
                                              np.array(smiles_2),
                                              np.array(Y),
                                              np.array(context),
                                              128, device=torch.device("cuda"))

    args.output_size = len(dti_data[0].targets)  ## target size
    args.num_tasks = 1
    args.train_data_size = len(single_data)

    return dti_data, single_data, comb_train_dataset, comb_valid_dataset, comb_test_dataset


def train(dti_data, single_data, comb_train_data, model, optimizer, scheduler, loss_func, loss_func_comb, args):
    model.train()
    single_data.shuffle()
    dti_data.shuffle()

    comb_loader = DataLoader(comb_train_data, batch_size=args.batch_size, shuffle=True)
    dataloader_iterator = iter(comb_loader)

    progress_bar = trange(0, len(comb_train_data), args.batch_size)
    for i in progress_bar:  # choose short lines between dti, single-agent and comb_train_data
        model.zero_grad()

        single_batch = MoleculeDataset(single_data[i:i + args.batch_size])
        dti_batch = MoleculeDataset(dti_data[i:i + args.batch_size])
        if len(single_batch) < args.batch_size:
            continue

        # DTI batch
        smiles, targets = dti_batch.smiles(), dti_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model.encoder.encode(smiles)[:, :targets.size(1)]
        dti_loss = loss_func(preds, targets)
        dti_loss = (dti_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        # single-agent batch
        smiles, targets = single_batch.smiles(), single_batch.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in targets]).cuda()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in targets]).cuda()
        preds = model(smiles, mode=0)
        single_loss = loss_func(preds, targets)
        single_loss = (single_loss * mask).sum() / mask.sum()
        smiles = targets = mask = None

        # comb batch
        try:
            sample = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(comb_loader)
            sample = next(dataloader_iterator)

        smiles_1, smiles_2, compounds_1, compounds_2, synergyScores, context, fp1, fp2 = sample
        compounds_1, compounds_2, synergyScores, context, fp1, fp2 = compounds_1.to(device), compounds_2.to(
            device), synergyScores.to(device), context.to(device), fp1.to(device), fp2.to(device)

        pre_synergy = model.combo_forward(smiles_1, smiles_2, compounds_1, compounds_2, context, fp1, fp2, mode=0)
        pre_synergy2 = model.combo_forward(smiles_1, smiles_2, compounds_1, compounds_2, context, fp1, fp2, mode=0)

        ce_loss = 0.5 * (loss_func_comb(pre_synergy, synergyScores.squeeze(1)) + loss_func_comb(pre_synergy2,
                                                                                                synergyScores.squeeze(
                                                                                                    1)))
        kl_loss = compute_kl_loss(pre_synergy, pre_synergy2)
        α = 5
        combo_loss = ce_loss + α * kl_loss

        loss = args.dti_lambda * dti_loss + args.single_lambda * single_loss + args.combo_lambda * combo_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'Loss': loss.item(),
                                  'dti_loss': dti_loss.item(),
                                  'single_loss': single_loss.item(),
                                  'comb_loss': combo_loss.item()})


def validate_new(data, model, args):
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()

    valid_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            smiles_1, smiles_2, compounds_1, compounds_2, Y, context, fp1, fp2 = batch
            compounds_1, compounds_2, Y, context, fp1, fp2 = compounds_1.to(device), compounds_2.to(device), Y.to(
                device), context.to(device), fp1.to(device), fp2.to(device)

            pre_synergy = model.combo_forward(smiles_1, smiles_2, compounds_1, compounds_2, context, fp1, fp2, mode=0)
            pre_synergy = torch.nn.functional.softmax(pre_synergy)[:, 1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)
            trues = torch.cat((trues, Y.view(-1, 1).cpu()), 0)

        y_pred = np.array(preds) > 0.5

        roc_auc = roc_auc_score(trues, preds)
        ACC = accuracy_score(trues, y_pred)
        F1 = f1_score(trues, y_pred, average='binary')
        Prec = precision_score(trues, y_pred, average='binary')
        Rec = recall_score(trues, y_pred, average='binary')
        ap = average_precision_score(trues, preds)

        return roc_auc


def run_training(args, save_dir):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dti_data, single_data, comb_train, comb_val, comb_test = prepare_data(args)

    model = DiseaseModel(args).cuda()
    loss_func = get_loss_func(args)  # for dti and single-agent task
    loss_func_comb = nn.CrossEntropyLoss()  # for synergy classification task (using binarized label)
    # loss_func_comb = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=args.train_data_size / args.batch_size * 2,
                                     t_total=args.train_data_size / args.batch_size * args.epochs
                                     )

    args.metric_func = get_metric_func(metric=args.metric)
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_epoch = 0

    for epoch in range(10):
        print(f'Epoch {epoch}')
        train(dti_data, single_data, comb_train, model,
              optimizer, scheduler, loss_func, loss_func_comb, args)

        val_scores = validate_new(comb_val, model, args)
        avg_val_score = np.nanmean(val_scores)
        print(f'Combo Validation {args.metric} = {avg_val_score:.4f}')

        # only save checkpoints when DTI prediction is accurate enough (after five epochs)
        if epoch >= 5 and (
                args.minimize_score and avg_val_score < best_score or not args.minimize_score and avg_val_score > best_score):
            best_score, best_epoch = avg_val_score, epoch
            save_checkpoint(os.path.join(save_dir, 'model.pt'), model, args=args)

    print(f'Loading model checkpoint from epoch {best_epoch}')
    ckpt_path = os.path.join(save_dir, 'model.pt')
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])

    test_scores = validate_new(comb_test, model, args)
    avg_test_scores = np.nanmean(test_scores)
    print(f'Test {args.metric} = {avg_test_scores:.4f}')

    return avg_test_scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dti_path', default="data/crc/dti.csv")
    parser.add_argument('--single_path', default="data/crc/single_agent_all.csv")
    parser.add_argument('--single_lambda', type=float, default=1)
    parser.add_argument('--combo_lambda', type=float, default=1)
    parser.add_argument('--dti_lambda', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--num_crc_targets', type=int, default=34)


    add_train_args(parser)
    args = parser.parse_args()
    args.dataset_type = 'classification'
    args.num_folds = 5

    modify_train_args(args)
    args.save_dir = "ckpt"
    print(args)

    all_test_scores = [0] * args.num_folds
    for i in range(0, args.num_folds):
        fold_dir = os.path.join(args.save_dir, f'fold_{i}')
        makedirs(fold_dir)
        args.seed = i
        all_test_scores[i] = run_training(args, fold_dir)

    all_test_scores = np.stack(all_test_scores, axis=0)
    mean, std = np.mean(all_test_scores, axis=0), np.std(all_test_scores, axis=0)
    print(f'{args.num_folds} fold average: {mean} +/- {std}')
