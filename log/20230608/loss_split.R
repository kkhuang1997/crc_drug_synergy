# 定义字符串切割函数
split_string <- function(str) {
  # 使用逗号作为分隔符切割字符串
  result <- strsplit(str, ",")
  # 返回切割后的字符串向量
  return(result[[1]])
}

loss$split <- apply(loss, 1, function(row) split_string(row["Epoch 0"]))
df <- apply(loss,2,as.character)
write.csv(df, "D://Desktop//loss_split.csv")

