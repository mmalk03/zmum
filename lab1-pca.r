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
