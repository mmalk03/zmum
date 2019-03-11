# Part 1
data <- read.table('SAheart.data', sep = ',', header = T)
data <- data[, -1]  # remove row_names
data.logit <- glm(chd ~ ., data = data, family = 'binomial')
summary(data.logit)

odds_ratio <- exp(coef(data.logit)['age'])  # iloraz szans

data.logit.aic <- step(data.logit, direction = 'backward', k = 2)  # AIC: 2 * log(alpha_hat) + p * 2
data.logit.bic <- step(data.logit, direction = 'backward', k = log(nrow(data)))  # BIC: 2 * log(alpha_hat) + p * log(n)

# Part 2
library('ggplot2')
eq_data <- read.table('earthquake.txt', sep = ' ', header = T)
ggplot(data = eq_data, aes(x = body, y = surface, col = popn)) + geom_point()
# it is clearly visible that the data is linearly separable - this is a problem for logistic regression
m1 <- glm(popn ~ ., data = eq_data, family = 'binomial')
summary(m1)

library('MASS')
l1 <- lda(popn ~ ., data = eq_data)
pred_lda <- predict(l1)
table(pred_lda$class, eq_data$popn)

q1 <- qda(popn ~ ., data = eq_data)
pred_qda <- predict(q1)
table(pred_qda$class, eq_data$popn)
