library(ISLR)
library(rpart)
library(rpart.plot)
set.seed(123)

# Load and split data
df <- ISLR::Default
df$default = ifelse(df$default == 'Yes', 1, 0)
df$student = ifelse(df$student == 'Yes', 1, 0)

sample_size <- floor(0.5 * nrow(df))
indices <- sample(seq_len(nrow(df)), size = sample_size)
train <- df[indices, ]
test <- df[-indices, ]

# ***** Task 1 *****

# glm - train and predict
df.logit <- glm(default ~ ., data = train, family = 'binomial')
glm_prob <- predict(df.logit, newdata = test, type = 'response')
glm_pred <- round(glm_prob)
glm_conf_mat <- table(test$default, glm_pred)
glm_accuracy <- sum(diag(glm_conf_mat)) / sum(glm_conf_mat)
paste0('GLM accuracy: ', glm_accuracy)

# tree - train and predict
tree <- rpart(as.factor(default) ~ ., data = train)
rpart.plot(tree)
tree_pred <- predict(tree, newdata = test, type = 'class')
tree_conf_mat <- table(test$default, tree_pred)
tree_accuracy <- sum(diag(tree_conf_mat)) / sum(tree_conf_mat)
paste0('Tree accuracy: ', tree_accuracy)

# true positive rates, precision and f1 score
glm_tpr <- glm_conf_mat[2, 2] / sum(glm_conf_mat[2,])
glm_prec <- glm_conf_mat[2, 2] / sum(glm_conf_mat[, 2])

tree_tpr <- tree_conf_mat[2, 2] / sum(tree_conf_mat[2,])
tree_prec <- tree_conf_mat[2, 2] / sum(tree_conf_mat[, 2])

glm_f1score <- 2 * glm_conf_mat[2, 2] / (sum(glm_conf_mat[, 2]) + sum(glm_conf_mat[2,]))
tree_f1score <- 2 * tree_conf_mat[2, 2] / (sum(tree_conf_mat[, 2]) + sum(tree_conf_mat[2,]))

# precision@k%
k <- 0.05  # or: k = mean(test$default)
glm_pred_atk <- ifelse(glm_prob > quantile(glm_prob, 1 - k), 1, 0)
glm_conf_mat_atk <- table(test$default, glm_pred_atk)
glm_prec_atk <- glm_conf_mat_atk[2, 2] / sum(glm_conf_mat_atk[, 2])
glm_f1score_atk <- 2 * glm_conf_mat_atk[2, 2] / (sum(glm_conf_mat_atk[, 2]) + sum(glm_conf_mat_atk[2,]))

# ***** Task 2 *****
glm_prob_sort <- sort(glm_prob)
ntest <- nrow(test)
tpr <- numeric(length(glm_prob_sort))
fpr <- numeric(length(glm_prob_sort))
for(i in 1 : length(glm_prob_sort)) {
  print(i)
  pred <- ifelse(glm_prob > glm_prob_sort[i], 1, 0)
  tab <- table(test$default, pred)
  if(ncol(tab) == 1) {
    tab <- cbind(tab, c(0, 0))
  }
  tpr[i] <- tab[2, 2] / sum(tab[2,])
  fpr[i] <- tab[1, 2] / sum(tab[1,])
}
plot(fpr, tpr, type = 'l', ylim = c(0, 1), xlim = c(0, 1), col = 'orange')
legend('bottomright', c('Logistic regression'), lwd = 2, lty = 1, col = 'orange')
