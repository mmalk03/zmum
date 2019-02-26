# 7% of 1s
# 93% of 0s
# in both train and test datasets

# During assessment:
#   * sort probabilities in descending order
#   * take 10% with biggest probabilities
#   * count the number of 1s in this set (sum(true_positive) / (sum(true_positive) + sum(false_positive)))

train <- read.table('~/Desktop/ZMUM/project/train.txt')
head(train)
length(unique(rownames(train)))
test <- read.table('~/Desktop/ZMUM/project/testx.txt')
head(test)
janlos <- read.table('~/Desktop/ZMUM/project/JANLOS.txt')
head(janlos)
