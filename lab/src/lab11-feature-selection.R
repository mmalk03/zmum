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

