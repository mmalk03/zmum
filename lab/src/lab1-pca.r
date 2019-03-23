# Task 1
usa_data_frame = data.frame(scale(USArrests))
cov1 <- cov(usa_data_frame)
eigen1 <- eigen(cov1)
scores <- as.matrix(usa_data_frame) %*% eigen1$vectors
sqrt(eigen1$values)
pca <- princomp(~., data=usa_data_frame)

eigen1$vectors
pca$loadings

scores
pca$scores

# Task 2
pca
vars <- pca$sdev^2
vars_perc <- vars / sum(vars)
plot(1:4, cumsum(vars_perc), type='b', col='orange', lwd=2, cex.lab=1.4, xlab="Comp", ylab="Principal component")
biplot(pca, col=c("black", "orange"), cex=c(0.8, 0.9), cex.lab=1.4)

# Task 3
install.packages("ISLR")
library(ISLR)
Hitters2 <- Hitters[, c(1:18, 20, 19)]
Hitters2 <- na.omit(Hitters2)
x <- model.matrix(Salary~., Hitters2)[, -1]
x_std <- data.frame(scale(x))
pca <- princomp(~., data=x_std)
plot(pca)

data_temp <- data.frame(pca$scores, Salary=Hitters2$Salary)
lm.fit1 <- lm(Salary~., data=data_temp)
summary(lm.fit1)
lm.fit2 <- lm(Salary~., data=Hitters2)
summary(lm.fit2)

p <- ncol(x)
pca.r2 <- numeric(p)
for (j in 1:p) {
  lm.fit <- lm(Salary ~ ., data = data_temp[, c(1:j, ncol(data_temp))])
  pca.r2[j] <- summary(lm.fit)$r.squared
}
plot(1:p, pca.r2, type = 'b', col = 'orange', lwd = 2, xlab = 'Variables', ylab = 'R2')

lm.fit2 <- lm(Salary~., data=Hitters2)
summary(lm.fit2)
tstat = abs(summary(lm.fit2)$coef[-1, 3])
order1 = order(tstat, decreasing = T)
lm.r2 <- numeric(p)
for (j in 1:p) {
  lm.fit <- lm(Salary~., data = Hitters2[, c(order1[1:j], ncol(data_temp))])
  lm.r2[j] <- summary(lm.fit)$r.squared
}

plot(1:p, pca.r2, type = 'b', col = 'orange', lwd = 2, xlab = 'Variables', ylab = 'R2', ylim = c(0, 0.6))
lines(1:p, lm.r2, col = 'blue', lwd = 2, type = 'b')
