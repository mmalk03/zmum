n <- 200
w <- 0.9
x <- w * rnorm(n, 5, 1) + (1 - w) * rnorm(n, 10, 1)
hist(x, breaks = 15, probability = TRUE)

x_lab <- seq(0, 10, 0.1)
y_lab <- dnorm(x_lab, 5.5, 0.9055385)
plot(x_lab, y_lab, type = 'l')

y_teor <- dnorm(sort(x), 5.5, 0.9055385)
points(sort(x), y_teor, col = 'blue', type = 'l')

empirical <- density(x, kernel = 'gaussian')
lines(empirical, type = 'l', col = 'red')

empirical2 <- density(x, kernel = 'gaussian', from = 2, to = 12, n = 512)
empirical2$x
y_teor2 <- dnorm(sort(empirical2$x), 5.5, 0.9055385)
mean((empirical2$y - y_teor2) ^ 2)

# Part 2
n <- 200
p <- 0.9
w <- rbinom(n, 1, p)
x <- w * rnorm(n, 5, 1) + (1 - w) * rnorm(n, 10, 1)
hist(x, breaks = 15, probability = TRUE)

y_teor <- w * dnorm(sort(x), 5, 1) + (1 - w) * dnorm(sort(x), 10, 1)
points(sort(x), y_teor, col = 'blue', type = 'l')

empirical <- density(x, kernel = 'gaussian')
lines(empirical, type = 'l', col = 'red')

empirical2 <- density(x, kernel = 'gaussian', from = 2, to = 12, n = 512)
empirical2$x
y_teor2 <- dnorm(sort(empirical2$x), 5.5, 0.9055385)
mean((empirical2$y - y_teor2) ^ 2)

# Task 2
library(MASS)
data(geyser)

x <- geyser$waiting
y <- geyser$duration

h1 <- bw.SJ(x)
h2 <- bw.SJ(y)

f.SJ <- kde2d(x, y, n = 100, h = c(h1, h2))
persp(f.SJ, col = 'orange', phi = 10, theta = -50, xlab = 'waiting', ylab = 'duration')
h1 <- 20 * h1
h2 <- 20 * h2
f.SJ <- kde2d(x, y, n = 100, h = c(h1, h2))
persp(f.SJ, col = 'orange', phi = 10, theta = -50, xlab = 'waiting', ylab = 'duration')

# Task 3
earth <- read.table('../data/earthquake.txt', header = T)
attach(earth)
den_equake <- density(body[popn == 'equake'], bw = 0.2)
den_explosn <- density(body[popn == 'explosn'], bw = 0.2)
plot(den_equake, type = 'l', col = 'red', ylim = c(0, 2))
lines(den_explosn, col = 'blue', ylim = c(0, 2))

p_equake <- sum(popn == 'equake') / nrow(earth)
p_explosn <- sum(popn == 'explosn') / nrow(earth)
plot(den_equake$x, p_equake * den_equake$y, type = 'l', col = 'red', ylim = c(0, 2))
lines(den_equake$x, p_explosn * den_equake$y, col = 'blue', ylim = c(0, 2))

f_equake <- kde2d(body[popn == 'equake'], surface[popn == 'equake'])
f_explosn <- kde2d(body[popn == 'explosn'], surface[popn == 'explosn'])
contour(f_equake, col = 'red')
contour(f_explosn, add = T, col = 'blue')
points(body[popn == 'equake'], surface[popn == 'equake'], pch = 19, col = 'red')
points(body[popn == 'explosn'], surface[popn == 'explosn'], pch = 19, col = 'blue')

persp(f_equake, col = 'red')
persp(f_explosn, col = 'blue')

# Task 4
df <- read.csv2('../data/winequality-white.csv')
df$quality[df$quality <= 5] <- 0
df$quality[df$quality > 5] <- 1
df$quality <- as.factor(df$quality)

indices <- sample(1:nrow(df), size = floor(2 * nrow(df) / 2))
train <- df[indices, ]
test <- df[-indices, ]

library(e1071)
lda_train <- lda(quality ~ ., data = train)
qda_train <- qda(quality ~ ., data = train)
nb_train <- naiveBayes(quality ~ ., data = train)

lda_predict <- predict(lda_train, test)$posterior[,2]
qda_predict <- predict(qda_train, test)$posterior[,2]
nb_predict <- predict(nb_train, test, type = 'raw')[,2]

library(ROCR)
labels <- test$quality

lda_poz <- prediction(lda_predict, 'tpr', 'fpr')
qda_poz <- prediction(qda_predict, 'tpr', 'fpr')
nb_poz <- prediction(nb_predict, 'tpr', 'fpr')

# AUC
auc_lda_poz <- performance(lda_poz, 'auc')
auc_qda_poz <- performance(qda_poz, 'auc')
auc_nb_poz <- performance(nb_poz, 'auc')

auc_lda_poz$y.values[[1]]
auc_qda_poz$y.values[[1]]
auc_nb_poz$y.values[[1]]
