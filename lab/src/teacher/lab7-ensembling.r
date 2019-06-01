###############################################
############ KOMITETY KLASYFIKATOROW ##########
###############################################


###### ZADANIE 1 #######

library(rpart)

bagging_drzew <- function(formula, data, B = 100, ...){
  drzewa <- vector("list", B)
  n <- nrow(data)
  for(i in 1:B){
    w <- sample(x = 1:n, size = n, replace = T)
    dane <- data[w,]
    drzewa[[i]] <- rpart(formula = formula, data = dane, ...)
  }
  return(drzewa)
}

drzewo <- function(formula, data, ...){
  rpart(formula = formula, data = data, ...)
}

library(foreign)
dane <- read.arff("https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/diabetes.arff")


# porownanie jakosci klasyfikacji baggingu drzew i pojedynczego drzewa 
n <- 20
dokl_bag <- dokl_drz <- numeric(n)
i <- 1
for(i in 1:n){
  w <- sample(1:nrow(dane), floor(2*nrow(dane)/3))
  train <- dane[w,]
  test <- dane[-w,]
  
  bagging <- bagging_drzew(as.formula("class ~ ."), data = train, B = 1000)
  drzewo <- rpart(class ~ ., data = train)
  
  pred_drzewo <- predict(drzewo, newdata = test, type = "class")
  
  pred_bag <- sapply(bagging, FUN = function(x){
    predict(x, newdata = test, type = "class")
  })
  
  pred_bag_2 <- apply(pred_bag, MARGIN = 1, FUN = function(x){
    ifelse(sum(x == "tested_negative") > sum(x == "tested_positive"), "tested_negative", 
           "tested_positive")
  })
  
  dokl_bag[i] <- sum(test$class == pred_bag_2)/nrow(test)
  dokl_drz[i] <- sum(test$class == pred_drzewo)/nrow(test)
}

mean(dokl_bag)
mean(dokl_drz)




###### ZADANIE 2 #######

library(randomForest)
library(rpart)
library(foreign)
library(adabag)

dane <- read.arff("https://storm.cis.fordham.edu/~gweiss/data-mining/weka-data/diabetes.arff")

bagging_drzew <- function(formula, data, B = 100, ...){
  drzewa <- vector("list", B)
  n <- nrow(data)
  for(i in 1:100){
    w <- sample(x = 1:n, size = n, replace = T)
    dane <- data[w,]
    drzewa[[i]] <- rpart(formula = formula, data = dane, ...)
  }
  return(drzewa)
}



ile <- 10
dokladnosc <- czulosc <- precyzja <- data.frame(drzewo = numeric(ile), bagging = numeric(ile), 
                                                boosting = numeric(ile), las = numeric(ile))

for(i in 1:ile){
  w <- sample(1:nrow(dane), size = floor(2*nrow(dane)/3))
  train <- dane[w,]
  test <- dane[-w,]
  
  bag <- bagging_drzew(formula = as.formula("class ~ ."), data = train, B = 100, method = "class")
  drz <- rpart(class ~ ., data = train, method = "class")
  las <- randomForest(class ~., data = train, ntree = 100)
  adaboost <- boosting(class ~., data = train, mfinal = 100)
  
  pred.drz <- predict(drz, newdata = test, type = "class")
  pred.bag <- sapply(bag, FUN = function(x){predict(x, newdata = test, type = "class")})
  pred.bag <- apply(pred.bag, MARGIN = 1, FUN = function(x){names(sort(table(x), decreasing = T))[1]})
  pred.las <- predict(las, newdata = test)
  pred.ada <- predict(adaboost, newdata = test)$class
  
  table.drz <- table(y = test$class, yy = pred.drz)
  table.bag <- table(y = test$class, yy = pred.bag)
  table.las <- table(y = test$class, yy = pred.las)
  table.ada <- table(y = test$class, yy = pred.ada)
  
  dokladnosc[i,1] <- sum(diag(table.drz))/sum(table.drz)
  dokladnosc[i,2] <- sum(diag(table.bag))/sum(table.bag)
  dokladnosc[i,3] <- sum(diag(table.las))/sum(table.las)
  dokladnosc[i,4] <- sum(diag(table.ada))/sum(table.ada)
  
  czulosc[i,1] <- table.drz[1,1]/sum(table.drz[1,])
  czulosc[i,2] <- table.bag[1,1]/sum(table.bag[1,])
  czulosc[i,3] <- table.las[1,1]/sum(table.las[1,])
  czulosc[i,4] <- table.ada[1,1]/sum(table.ada[1,])
  
  precyzja[i,1] <- table.drz[1,1]/sum(table.drz[,1])
  precyzja[i,2] <- table.bag[1,1]/sum(table.bag[,1])
  precyzja[i,3] <- table.las[1,1]/sum(table.las[,1])
  precyzja[i,4] <- table.ada[1,1]/sum(table.ada[,1])
}

#wyciagam srednie wynikow
apply(dokladnosc, MARGIN = 2, mean)
apply(czulosc, MARGIN = 2, mean)
apply(precyzja, MARGIN = 2, mean)



# porownanie krzywych ROC

w <- sample(1:nrow(dane), size = floor(2*nrow(dane)/3))
train <- dane[w,]
test <- dane[-w,]

drzewo <- rpart(class ~ ., data = train)
las <- randomForest(class ~ ., data = train, ntree = 100)
bag <- bagging_drzew(class ~ ., data = train, B = 100)
boos <- boosting(class ~., data = train, mfinal = 100)

pstwa_drzewo_poz <- predict(drzewo, newdata = test)[,"tested_positive"]
pstwa_las_poz <- predict(las, newdata = test, type = "prob")[,"tested_positive"]
pstwa_bag_poz <- sapply(bag, FUN = function(x){predict(x, newdata = test, type = "class")})
pstwa_bag_poz <- apply(pstwa_bag_poz, MARGIN = 1, FUN = function(x){sum(x == "tested_positive")/length(x)})
pstwa_boos_poz <- predict(boos, test)$prob[,2]


library(ROCR)
Labels <- ifelse(test$class == "tested_positive", 1, 0)


Pred_drzewo_poz <- prediction(pstwa_drzewo_poz, Labels)
Pred_las_poz <- prediction(pstwa_las_poz, Labels)
Pred_bag_poz <- prediction(pstwa_bag_poz, Labels)
Pred_boos_poz <- prediction(pstwa_boos_poz, Labels)

perf_drzewo_poz <- performance(Pred_drzewo_poz, "tpr", "fpr")
perf_las_poz <- performance(Pred_las_poz, "tpr", "fpr")
perf_bag_poz <- performance(Pred_bag_poz, "tpr", "fpr")
perf_boos_poz <- performance(Pred_boos_poz, "tpr", "fpr")

par(mfrow = c(2,2))
plot(perf_drzewo_poz)
plot(perf_las_poz)
plot(perf_bag_poz)
plot(perf_boos_poz)
par(mfrow = c(1,1))

# parametr AUC
auc_drzewo_poz <- performance(Pred_drzewo_poz, "auc")
auc_las_poz <- performance(Pred_las_poz, "auc")
auc_bag_poz <- performance(Pred_bag_poz, "auc")
auc_boos_poz <- performance(Pred_boos_poz, "auc")

auc_drzewo_poz@y.values[[1]]
auc_las_poz@y.values[[1]]
auc_bag_poz@y.values[[1]]
auc_boos_poz@y.values[[1]]




###### ZADANIE 3 #######
library("ISLR")
library("MASS")
library("mboost")
library("randomForest")
library("rpart")
data(Boston)

set.seed(123)
train_id =  sample(1:nrow(Boston),300)
test_id = setdiff(1:nrow(Boston),train_id)

train = Boston[train_id,]
test = Boston[test_id,]
B = 50

err1 = numeric(B)
err2 = numeric(B)
err3 = numeric(B)

for(b in 1:B){
cat("Simulation ",b," out of ",B,"\n")
  
del = sample(1:nrow(train),0.1*nrow(train))  
train1 = train[-del,]  

model1 = randomForest(medv~.,data=train1)
model2 = rpart(medv~.,data=train1)
model3= glmboost(medv~.,data=train1,family=Gaussian())

err1[b] = sqrt(sum((predict(model1,test)-test$medv)^2))
err2[b] = sqrt(sum((predict(model2,test)-test$medv)^2))
err3[b] = sqrt(sum((predict(model3,test)-test$medv)^2))
}


boxplot(err1,err2,err3,names=c("randomForest","rpart","glmboost"),col="orange")
cat("variance for randomForest=",var(err1))
cat("variance for rpart=",var(err2))
cat("variance for glmboost=",var(err3))
