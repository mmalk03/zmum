###############################################
######### METODY OCENY KLASYFIKATOROW #########
###############################################


###### ZADANIE 1 #######

#Wczytanie danych:
library(ISLR)
data(Default)
Default$student=ifelse(Default$student=="Yes",1,0)
Default$default=ifelse(Default$default=="Yes",1,0)

n = nrow(Default)
train_sample = sample(1:n,0.5*n)
test_sample = setdiff(1:n,train_sample)
train = Default[train_sample,]
test = Default[test_sample,]

#Model + predykcja:
model = glm(default~.,data=train,family="binomial")
prob = predict(model,test,type="response")
pred = ifelse(prob>0.5,1,0)
tab = table(test$default,pred)

#Obliczenie miar:
acc = sum(diag(tab))/sum(tab)
precision = tab[2,2]/sum(tab[,2])
recall = tab[2,2]/sum(tab[2,])
fmeasure = 2 * precision *recall/(precision +recall)

k = 0.1
pred = ifelse(prob>quantile(prob,1-k),1,0)
tab = table(test$default,pred)
precision = tab[2,2]/sum(tab[,2])



###### ZADANIE 2 #######
library(ISLR)
data(Default)
Default$student=ifelse(Default$student=="Yes",1,0)
Default$default=ifelse(Default$default=="Yes",1,0)

n = nrow(Default)
train_sample = sample(1:n,0.5*n)
test_sample = setdiff(1:n,train_sample)
train = Default[train_sample,]
test = Default[test_sample,]

#Model + predykcja:
model = glm(default~.,data=train,family="binomial")
prob = predict(model,test,type="response")


#Rysowanie krzewej ROC:
th = sort(c(prob))  
ntest = nrow(test)
TPR = numeric(length(th))  
FPR = numeric(length(th))

for(i in 1:length(th)){
  print(i)
  pred = ifelse(prob>th[i],1,0)
  tab = table(test$default,pred)
  
  TPR[i] = tab[2,2]/sum(tab[2,]) 
  FPR[i] = tab[1,2]/sum(tab[1,])
}

pdf(file="ROC.pdf")
plot(FPR,TPR,type="l",ylim=c(0,1),xlim=c(0,1),xlab="FPR",ylab="TPR",main="Krzywa ROC",lwd=2,col="orange")
legend("bottomright",c("Regresja logistyczna"),lwd=2,lty=1,col="orange")
dev.off()




