library('ggplot2')
library('e1071')

plot_points <- function(x, y) {
  ggplot(data = data.frame(x = x, y = y), aes(x = x.1, y = x.2, col = as.factor(y))) +
    geom_point()
}

y <- c(rep(1, 500), rep(0, 500))
x <- matrix(0, ncol = 2, nrow = 1000)

x[1:500, 1] <- runif(500, -1, 1)
x[501:1000, 1] <- runif(500, -2, 2)

x[1:500, 2] <- sample(c(-1, 1), size = 500, replace = T, prob = c(0.5, 0.5)) * sqrt(1 - x[1:500, 1] ^ 2)
x[501:1000, 2] <- sample(c(-1, 1), size = 500, replace = T, prob = c(0.5, 0.5)) * sqrt(4 - x[501:1000, 1] ^ 2)

x[, 2] <- x[, 2] + rnorm(1000, 0, 0.1)

plot(x[, 1], x[, 2], col = as.factor(y))
plot_points(x, y)

m1 <- svm(x, as.factor(y), kernel = 'linear')
y_hat1 <- predict(m1, x)
plot_points(x, y_hat1)

m2 <- svm(x, as.factor(y), kernel = 'radial')
y_hat2 <- predict(m2, x)
plot_points(x, y_hat2)

m3 <- svm(x, as.factor(y), kernel = 'polynomial')
y_hat3 <- predict(m3, x)
plot_points(x, y_hat3)

