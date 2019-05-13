# Task 1
n <- 100
p <- 1000
L <- 100

max_cor <- function(n, p, L) {
  corr <- numeric(L)
  for (k in 1:L) {
    corr1 <- numeric(p)
    y <- rnorm(n)
    x <- matrix(rnorm(n * p), ncol = p)
    for (j in 1:p) {
      corr1[j] <- cor(y, x[, j])
    }
    corr[k] <- max(corr1)
  }
  return (corr)
}
max1 <- max_cor(100, 1000, 100)
max2 <- max_cor(100, 10000, 100)
plot(density(max1), col = 'red')
plot(density(max2), col = 'blue')

# Task 2
library('FSelector')

# Part 1
L <- 50
sigmaSeq <- seq(from = 0, to = 5, by = 0.1)
corr <- numeric(length(sigmaSeq))
mi <- numeric(length(sigmaSeq))
i <- 1
for (sigma in sigmaSeq) {
  print(sigma)
  corr1 <- numeric(L)
  mi1 <- numeric(L)
  for (sym in 1:L) {
    x <- runif(100, 0, 1)
    e <- rnorm(100, 0, sigma)
    y <- 2 * x + e
    corr1[sym] <- cor(y, x)
    mi1[sym] <- as.numeric(information.gain(data.frame(x, y)))
  }
  corr[i] <- mean(corr1)
  mi[i] <- mean(mi1)
  i <- i + 1
}

plot(sigmaSeq, corr, col = 'red', type = 'l', ylim = c(0, max(c(corr, mi))))
lines(sigmaSeq, mi, col = 'blue')

# Part 3
L <- 50
sigmaSeq <- seq(from = 0, to = 5, by = 0.1)
corr <- numeric(length(sigmaSeq))
mi <- numeric(length(sigmaSeq))
i <- 1
for (sigma in sigmaSeq) {
  print(sigma)
  corr1 <- numeric(L)
  mi1 <- numeric(L)
  for (sym in 1:L) {
    x <- runif(100, -1, 1)
    e <- rnorm(100, 0, sigma)
    y <- x ^ 2 + e
    corr1[sym] <- cor(y, x)
    mi1[sym] <- as.numeric(information.gain(data.frame(x, y)))
  }
  corr[i] <- mean(corr1)
  mi[i] <- mean(mi1)
  i <- i + 1
}

plot(sigmaSeq, corr, col = 'red', type = 'l', ylim = c(0, max(c(corr, mi))))
lines(sigmaSeq, mi, col = 'blue')

# Task 3
library('glmnet')

n <- 100
p <- 1000
true <- c(1, 2, 3)
betas <- c(1, 1, 1, rep(0, p - 3))
x <- matrix(rnorm(n * p), ncol = p)
p <- 1 / (1 + exp(-x %*% betas))
y <- rbinom(n, 1, p)

cv.glmnet1 <- cv.glmnet(x, y, family = 'binomial')
lambda_opt <- cv.glmnet1$lambda.1se
model <- glmnet(x, y, family = 'binomial', lambda = lambda_opt)
coefficients <- model$beta

est_true1 <- which(coefficients != 0)
recall1 <- length(intersect(true, est_true1)) / length(true)
prec1 <- length(intersect(true, est_true1)) / length(est_true1)

d <- data.frame(x, y)
biggest <- formula(glm(y ~ ., data = d))
min.model <- glm(y ~ 1, data = d)
m2 <- step(min.model, data = d, direction = 'forward', scope = biggest, k = log(n))

library('hdi')
m3 <- multi.split(x, y)

# Task 4
n <- 200
p <- 20
B <- 100
coefs <- list()
j <- 0
for (p in c(5, 10, 20, 40, 50)) {
  coef1 <- numeric(B)
  for (k in 1:B) {
    true <- c(1, 2, 3)
    x <- matrix(rnorm(n * p), nrow = n, ncol = p)
    b <- numeric(p)
    b[true] <- 1
    eta <- x %*% b
    probs <- 1 / (1 + (exp(-eta)))
    y <- rbinom(n, 1, probs)
    d <- data.frame(x, y)
    m <- glm(y ~ ., data = d)
    coef1[k] <- m$coef[2]
  }
  j <- j + 1
  coefs[[j]] <- coef1
}
sapply(coefs, var)
boxplot(coefs)
