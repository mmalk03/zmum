# Lab 8
library('extraTrees')
library('xgboost')
library('rpart')
library('ROCR')
library('ipred')
library('randomForest')
library('adabag')
library('gbm')

sa <- read.table('../data/SAheart.data', h = T, row.names = 1, sep = ',')
sa$famhist <- ifelse(sa$famhist == 'Present', 1, 0)
sa$chd <- as.factor(sa$chd)

train_size <- floor(0.80 * nrow(sa))
train_indices <- sample(seq_len(nrow(sa)), size = train_size)
sa_train <- sa[train_indices, ]
sa_test <- sa[-train_indices, ]

# tree
m1 <- rpart(chd ~ ., data = sa_train)
pred <- predict(m1, sa_test)[, 2]
pred_all <- prediction(pred, sa_test$chd)
perf <- performance(pred_all, measure = 'auc')
perf@y.values
roc_ROCR1 <- performance(pred_all, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR1, main = 'ROC curve', col = 'orange')
abline(0, 1)

# bagging
m2 <- bagging(chd ~ ., data = sa_train)
pred2 <- predict(m2, sa_test)$prob[, 2]
pred_all2 <- prediction(pred2, sa_test$chd)
perf2 <- performance(pred_all2, measure = 'auc')
perf2@y.values
roc_ROCR2 <- performance(pred_all2, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR2, add = TRUE, col = 'blue')

# boosting
m3 <- boosting(chd ~ ., data = sa_train)
pred3 <- predict(m3, sa_test)$prob[, 2]
pred_all3 <- prediction(pred3, sa_test$chd)
perf3 <- performance(pred_all3, measure = 'auc')
perf3@y.values
roc_ROCR3 <- performance(pred_all3, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR3, add = TRUE, col = 'green')

# random forest
m4 <- randomForest(chd ~ ., data = sa_train)
pred4 <- predict(m4, sa_test, type = 'prob')[, 2]
pred_all4 <- prediction(pred4, sa_test$chd)
perf4 <- performance(pred_all4, measure = 'auc')
perf4@y.values
roc_ROCR4 <- performance(pred_all4, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR4, add = TRUE, col = 'red')

# gradient boosting
m5 <- gbm(chd ~ ., data = sa_train, distribution = 'multinomial')
pred5 <- matrix(predict(m5, sa_test, n.trees = 100, type = 'response'), ncol = 2)[, 2]
pred_all5 <- prediction(pred5, sa_test$chd)
perf5 <- performance(pred_all5, measure = 'auc')
perf5@y.values
roc_ROCR5 <- performance(pred_all5, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR5, add = TRUE, col = 'purple')

# xgboost
# TODO something wrong is here in xgboost
m6 <- xgboost(
  data = as.matrix(sa_train[, -10]),
  label = as.numeric(as.vector(sa_train[, 10])),
  objective = 'binary:logistic',
  watchlist = list(train = dtrain, eval = dtest),
  max.depth = 2,
  eta = 1,
  nthread = 2,
  nrounds = 100)
pred6 <- predict(m6, as.matrix(sa_test[, -10]))
pred_all6 <- prediction(pred6, sa_test$chd)
perf6 <- performance(pred_all6, measure = 'auc')
perf6@y.values
roc_ROCR6 <- performance(pred_all6, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR6, add = TRUE, col = 'purple')

# extremely random trees
m7 <- extraTrees(x = sa_train[, -10], y = sa_train$chd)
pred7 <- predict(m7, as.matrix(sa_test[, -10]), probability = TRUE)[, 2]
pred_all7 <- prediction(pred7, sa_test$chd)
perf7 <- performance(pred_all7, measure = 'auc')
perf7@y.values
roc_ROCR7 <- performance(pred_all7, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR7, add = TRUE, col = 'purple')
