library('rpart')
library('rpart.plot')

# Task 1
sa_heart_df <- read.table('SAheart.data', sep = ',', header = T, row.names = 1)
# a
tree <- rpart(as.factor(chd) ~ ., data = sa_heart_df, cp = 0.01, minsplit = 5)
# b - using rpart.plot
rpart.plot(tree)
print(summary(tree), digits = 4)
# b - manual visualization
par(mar = c(0, 1, 0, 1))
plot(tree)
text(tree)
# c
test_obs <- as.data.frame(t(apply(sa_heart_df[, -5], 2, mean)))
famhist <- names(which(table(sa_heart_df[, 5]) == max(table(sa_heart_df[, 5]))))
test_obs <- cbind(test_obs, famhist)
test_obs <- test_obs[, -9]
predict(tree, newdata = test_obs, type = 'class')
# d
plotcp(tree)
tree$cptable
z <- prune.rpart(tree, cp = 0.04)
rpart.plot(z)
tree2 <- rpart(as.factor(chd) ~ ., data = sa_heart_df, cp = 0.01,
               minsplit = 5, parms = list(split = 'information'))
rpart.plot(tree2)

# Task 3
fitness_df <- read.table('fitness.txt', h = TRUE)
# a
tree3 <- rpart(Oxygen ~ ., data = fitness_df, cp = 0.01, minsplit = 2)
rpart.plot(tree3)
# b
# Oxygen intake is highest for runners with RunTime < 8.9 and RunPulse >= 161
# c
x0 <- as.data.frame(t(apply(fitness_df, 2, median)))
x0 <- x0[, -3]
predict(tree3, newdata = x0)
# d
# complexity cost
plotcp(tree3)
tree3$cptable
z <- prune.rpart(tree3, cp = 0.04)
rpart.plot(z)
# 1SE
z <- prune.rpart(tree3, cp = 0.1)
rpart.plot(z)
# e
tree4 <- rpart(Oxygen ~ ., data = fitness_df[, c(1, 3, 4)], cp = 0.02, minsplit = 2)
n <- 100
age_seq <- seq(min(fitness_df$Age) - 1, max(fitness_df$Age) + 1, length.out = n)
runtime_seq <- seq(round(min(fitness_df$RunTime)), round(max(fitness_df$RunTime)) + 1, length.out = n)
newdata <- expand.grid(Age = age_seq, RunTime = runtime_seq)
newdata$z <- predict(tree4, newdata = newdata)
persp(age_seq, runtime_seq, matrix(newdata$z, n), theta = 50, phi = 30, expand = 0.5,
      col = 'lightblue', zlab = 'Oxygen prediction')

