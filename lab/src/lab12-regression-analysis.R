# Task 1
data <- read.table('../data/realest.txt', header=TRUE)
lm1 <- lm(Price ~ ., data = data)
summary(lm1)$r.squared
summary(lm1)

lm2 <- lm(Price ~ Bedroom, data = data)
summary(lm2)$r.squared
summary(lm2)

new_house <- matrix(c(3, 1500, 8, 40, 1000, 2, 1, 0), nrow = 1)
colnames(new_house) <- names(lm1$coefficients)[-1]
predict(lm1, data.frame(new_house))

# Task 2
library(car)
attach(USPop)
USPop
plot(year, population, type = 'b')
time <- (year - min(year)) / 10

m1 <- nls(
  population ~ beta1 / (1 + exp(beta2 + beta3 * time)),
  start = list(beta1 = 350, beta2 = 4.5, beta3 = -0.3),
  trace = T
)
m1

plot(year, population)
lines(year, fitted.values(m1), lwd = 2)

predict(m1, list(time = 22.5))

# Task 3
