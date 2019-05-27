# Task 1
library(earth)

hinge_r <- function(x, t) {
  pmax(0, x - t)
}

n <- 1000
p <- 50
x <- matrix(rnorm(n * p), nrow = n, ncol = p)
y <- hinge_r(x[, 1], 0) + hinge_r(x[, 1], 1) + rnorm(n, sd = 0.1)

d <- data.frame(y, x)
model <- earth(y ~ ., data = d, degree = 2, trace = 3, penalty = 0)
summary(model, digits = 2, style = 'pmax')

or1 <- order(x[, 1])
plot(x[or1, 1], y[or1])
lines(x[or1, 1], model$fitted.values[or1], col = 'red')

plot(model$fitted.values[or1], y[or1])

plot(model)

# Task 2
