###############################################
############ Selekcja zmiennych ###############
###############################################


###### ZADANIE 1 #######

L = 200 #liczba symulacji
n = 100


max_cor = function(n,p,L){
  maksima = numeric(L)
  
for(i in 1:L){
  cat("Simulation",i,"out of",L,"\n")
  y = rnorm(n)
  x = matrix(0,nrow=n,ncol=p)
  for(j in 1:p) x[,j] = rnorm(n)
  cor1 = numeric(p)
  for(j in 1:p) cor1[j] = cor(y,x[,j])
  maksima[i] = max(cor1)
}
return(maksima)  
}

maksima1 = max_cor(n,p=1000,L=50)
maksima2 = max_cor(n,p=10000,L=50)
maksima3 = max_cor(n,p=50000,L=50)

  
plot(density(maksima1),col="red")  
lines(density(maksima2),col="blue")  
lines(density(maksima3),col="orange")  
  
boxplot(maksima1,maksima2,maksima3)


###### ZADANIE 2 #######
install.packages("FSelector")
library(FSelector)

#Przyklad 1:
L =100
sigmaSeq = seq(from=0,to=5,by=0.1)
corr = numeric(length(sigmaSeq))
mi = numeric(length(sigmaSeq))
i=1
for(sigma in sigmaSeq){
  print(sigma)
corr1 = numeric(L)
mi1= numeric(L)
for(sym in 1:L){
x = runif(100,0,1)
e = rnorm(100,0,sigma)
y = 2*x + e
corr1[sym] = cor(y,x)
mi1[sym]=as.numeric(information.gain(data.frame(x,y)))
}
corr[i] = mean(corr1)
mi[i] = mean(mi1)
i=i+1
}
plot(sigmaSeq,corr,lwd=2,col="red",type="l",ylim=c(0,max(c(corr,mi))),xlab="sigma",ylab="cor/mi")
lines(sigmaSeq,mi,lwd=2,col="blue")


#Przyklad 2:
L =100
sigmaSeq = seq(from=0,to=5,by=0.1)
corr = numeric(length(sigmaSeq))
mi = numeric(length(sigmaSeq))
i=1
for(sigma in sigmaSeq){
  print(sigma)
  corr1 = numeric(L)
  mi1= numeric(L)
  for(sym in 1:L){
    x = runif(100,0,1)
    e = rnorm(100,0,sigma)
    y = sqrt(x) + e
    corr1[sym] = cor(y,x)
    mi1[sym]=as.numeric(information.gain(data.frame(x,y)))
  }
  corr[i] = mean(corr1)
  mi[i] = mean(mi1)
  i=i+1
}
plot(sigmaSeq,corr,lwd=2,col="red",type="l",ylim=c(0,max(c(corr,mi))),xlab="sigma",ylab="cor/mi")
lines(sigmaSeq,mi,lwd=2,col="blue")


#Przyklad 3:
L =50
sigmaSeq = seq(from=0,to=5,by=0.1)
corr = numeric(length(sigmaSeq))
mi = numeric(length(sigmaSeq))
i=1
for(sigma in sigmaSeq){
  print(sigma)
  corr1 = numeric(L)
  mi1= numeric(L)
  for(sym in 1:L){
    x = runif(100,-1,1)
    e = rnorm(100,0,sigma)
    y = x^2 + e
    corr1[sym] = cor(y,x)
    mi1[sym]=as.numeric(information.gain(data.frame(x,y)))
  }
  corr[i] = mean(corr1)
  mi[i] = mean(mi1)
  i=i+1
}
plot(sigmaSeq,corr,lwd=2,col="red",type="l",ylim=c(0,max(c(corr,mi))),xlab="sigma",ylab="cor/mi")
lines(sigmaSeq,mi,lwd=2,col="blue")

#Przyklad 4:
L =50
sigmaSeq = seq(from=0,to=5,by=0.1)
corr = numeric(length(sigmaSeq))
mi = numeric(length(sigmaSeq))
i=1
for(sigma in sigmaSeq){
  print(sigma)
  corr1 = numeric(L)
  mi1= numeric(L)
  for(sym in 1:L){
    x = runif(100,0,6)
    e = rnorm(100,0,sigma)
    y = sin(x) + e
    corr1[sym] = cor(y,x)
    mi1[sym]=as.numeric(information.gain(data.frame(x,y)))
  }
  corr[i] = mean(corr1)
  mi[i] = mean(mi1)
  i=i+1
}
plot(sigmaSeq,corr,lwd=2,col="red",type="l",ylim=c(min(c(corr,mi)),max(c(corr,mi))),xlab="sigma",ylab="cor/mi")
lines(sigmaSeq,mi,lwd=2,col="blue")


###### ZADANIE 3 #######
library(glmnet)

n = 1000
p = 500
true = c(1,2,3) # indeksy istotnych zmiennych:
x = matrix(0,nrow=n,ncol=p)
for(j in 1:p)x[,j]=rnorm(n)
b=numeric(p)
b[1:3] = 1
eta = x%*%b
probs = exp(eta)/(1+exp(eta))
y = rbinom(n,1,probs)
d = data.frame(x,y)
model = glmnet(x,y,family="binomial")
#Wybor optymalnego parametru lambda:
cv.glmnet1 = cv.glmnet(x,y,family="binomial")
lambda_opt = cv.glmnet1$lambda.1se 
wspolczynniki = model$beta[,which(model$lambda==lambda_opt)]
#Ktore zmienne wybralismy:
est_true = which(wspolczynniki!=0) 


recall = length(intersect(true,est_true))/length(true)
prec = length(intersect(true,est_true))/length(est_true)

print(recall)
print(prec)


###### ZADANIE 4 #######

n = 200
p = 50
coef1 = numeric(100)
for(k in 1:100){
cat("Simulation",k,"\n")  
true = c(1,2,3) # indeksy istotnych zmiennych:
x = matrix(0,nrow=n,ncol=p)
for(j in 1:p)x[,j]=rnorm(n)
b=numeric(p)
b[1:3] = 1
eta = x%*%b
probs = exp(eta)/(1+exp(eta))
y = rbinom(n,1,probs)
d = data.frame(x,y)
model = glm(y~.,data=d,family="binomial")
coef1[k] = coef(model)[2]
}
