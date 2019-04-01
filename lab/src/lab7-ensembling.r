library('rpart')
library('ROCR')

# Task 1
wine <- read.csv2('../data/winequality-red.csv', dec = '.')
unique(wine$quality)
wine$quality[wine$quality <= 5] <- 0
wine$quality[wine$quality > 5] <- 1
wine$quality <- as.factor(wine$quality)

samp <- sample(1:nrow(wine), 2 * nrow(wine) / 3)
wine_train <- wine[samp, ]
wine_test <- wine[-samp, ]

m1 <- rpart(quality ~ ., data = wine_train)
pred <- predict(m1, wine_test)[, 2]
pred_all <- prediction(pred, wine_test$quality)
perf <- performance(pred_all, measure = 'auc')
perf@y.values

bagging_pred <- function(num_class, data_train, data_test) {
  pred_mean <- numeric(nrow(data_test))
  num_obs <- nrow(data_train)
  for (i in 1 : num_class) {
    samp <- sample(1 : num_obs, replace = TRUE)
    data_boot <- data_train[samp, ]
    m <- rpart(quality ~ ., data = data_boot)
    pred <- predict(m, data_test)[, 2]
    pred_mean <- pred_mean + pred
  }
  pred_mean <- pred_mean / num_class
  return (pred_mean)
}

pred2 <- bagging_pred(100, wine_train, wine_test)
pred_all2 <- prediction(pred2, wine_test$quality)
perf2 <- performance(pred_all2, measure = 'auc')
perf2@y.values

# Task 2
library('ipred')
library('randomForest')
library('adabag')
library('gbm')

wine <- read.csv2('../data/winequality-white.csv', dec = '.')
unique(wine$quality)
wine$quality[wine$quality <= 5] <- 0
wine$quality[wine$quality > 5] <- 1
wine$quality <- as.factor(wine$quality)

samp <- sample(1:nrow(wine), 2 * nrow(wine) / 3)
wine_train <- wine[samp, ]
wine_test <- wine[-samp, ]

# standard tree
m1 <- rpart(quality ~ ., data = wine_train)
pred <- predict(m1, wine_test)[, 2]
pred_all <- prediction(pred, wine_test$quality)
perf <- performance(pred_all, measure = 'auc')
perf@y.values
roc_ROCR1 <- performance(pred_all, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR1, main = 'ROC curve', col = 'orange')
abline(0, 1)

# bagging
m2 <- bagging(quality ~ ., data = wine_train)
pred2 <- predict(m2, wine_test)$prob[, 2]
pred_all2 <- prediction(pred2, wine_test$quality)
perf2 <- performance(pred_all2, measure = 'auc')
perf2@y.values
roc_ROCR2 <- performance(pred_all2, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR2, add = TRUE, col = 'blue')

# boosting
m3 <- boosting(quality ~ ., data = wine_train)
pred3 <- predict(m3, wine_test)$prob[, 2]
pred_all3 <- prediction(pred3, wine_test$quality)
perf3 <- performance(pred_all3, measure = 'auc')
perf3@y.values
roc_ROCR3 <- performance(pred_all3, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR3, add = TRUE, col = 'green')

# random forest
m4 <- randomForest(quality ~ ., data = wine_train)
pred4 <- predict(m4, wine_test, type = 'prob')[, 2]
pred_all4 <- prediction(pred4, wine_test$quality)
perf4 <- performance(pred_all4, measure = 'auc')
perf4@y.values
roc_ROCR4 <- performance(pred_all4, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR4, add = TRUE, col = 'red')

# gradient boosting
m5 <- gbm(quality ~ ., data = wine_train, distribution = 'multinomial')
pred5 <- matrix(predict(m5, wine_test, n.trees = 100, type = 'response'), ncol = 2)[, 2]
pred_all5 <- prediction(pred5, wine_test$quality)
perf5 <- performance(pred_all5, measure = 'auc')
perf5@y.values
roc_ROCR5 <- performance(pred_all5, measure = 'tpr', x.measure = 'fpr')
plot(roc_ROCR5, add = TRUE, col = 'purple')

# Task 3
library('ISLR')
library('MASS')
library('mboost')

samp <- sample(1:nrow(Boston), 2 * nrow(Boston) / 3)
boston_train <- Boston[samp, ]
boston_test <- Boston[-samp, ]
B <- 100
err1 <- numeric(b)
err2 <- numeric(b)
err3 <- numeric(b)
for (b in 1 : B) {
  cat('Simulation ', b, ' out of ', B, '\n')
  del <- sample(1 : nrow(boston_train), 0.1 * nrow(boston_train))
  boston_train1 <- boston_train[-del, ]
  m1 <- rpart(medv ~ ., data = boston_train1)
  m2 <- randomForest(medv ~ ., data = boston_train1)
  m3 <- glmboost(medv ~ ., data = boston_train1, family = Gaussian())
  err1[b] <- sqrt(sum((boston_test$medv - predict(m1, boston_test)) ^ 2))
  err2[b] <- sqrt(sum((boston_test$medv - predict(m2, boston_test)) ^ 2))
  err3[b] <- sqrt(sum((boston_test$medv - predict(m3, boston_test)) ^ 2))
}
boxplot(err1, err2, err3, names = c('rpart', 'randomForest', 'glmboost'), col = 'orange')
mean(err1)
var(err1)
mean(err2)
var(err2)
mean(err3)
var(err3)
