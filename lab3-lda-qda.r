n <- 1000
x <- runif(n, -1, 1)
y <- rep(0, n)
for (i in 1:n) {
  y[i] <- runif(1, -sqrt(1 - x[i] ^ 2), sqrt(1 - x[i] ^ 2))
}
plot(x, y)
x1 <- x[x^2 + y^2 < 0.5]
y1 <- y[x^2 + y^2 < 0.5]
x2 <- x[x^2 + y^2 >= 0.5]
y2 <- y[x^2 + y^2 >= 0.5]
plot(x2, y2, col = 'red')
points(x1, y1, col = 'blue')

dane <- data.frame(x1 = x, x2 = y, y = ifelse(x^2 + y^2 < 0.5, 0, 1))

library('MASS')
q1 <- qda(y~., data = dane)
pred1 <- predict(q1)
table(pred1$class, dane$y)

q2 <- qda(y ~ x1^2 + x1*x2, data = dane)
pred2 <- predict(q2)
table(pred2$class, dane$y)

dane1 <- data.frame(x1, y1)
dane2 <- data.frame(x2, y2)

cov1 <- cov(dane1)
cov2 <- cov(dane2)

cov1 <- solve(cov1)
cov2 <- solve(cov2)
m1 <- apply(dane1, 2, mean)
m2 <- apply(dane2, 2, mean)
pi1 <- length(x1) / length(x)
pi2 <- length(x2) / length(x)
dane3 <- matrix(c(x, y), ncol = 2)

# TODO to be continued...
