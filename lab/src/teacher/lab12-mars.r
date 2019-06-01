###############################################
############ Metoda MARS ######################
###############################################


###### ZADANIE 1 #######
#Pomocnicze funkcje:
hinge_r = function(x,t){
  pmax(0,x-t)
}

hinge_l = function(x,t){
  pmax(0,-(x-t))
}

sim = "3"

if(sim=="1"){
  # Example 1:
  n = 1000
  p = 50
  X = matrix(0,ncol=p,nrow=n)
  for(j in 2:p) X[,j] = rnorm(n)
  X[,1] = runif(n,0,4)
  y = sqrt(X[,1])  + rnorm(n,sd=0.1)
}
if(sim=="2"){
  # Example 2:
  n = 1000
  p = 50
  X = matrix(0,ncol=p,nrow=n)
  for(j in 1:p) X[,j] = rnorm(n)
  y = X[,1]^2  + rnorm(n,sd=0.1)
}
if(sim=="3"){
  #Example 3:
  n = 1000
  p = 50
  X = matrix(0,ncol=p,nrow=n)
  for(j in 1:p) X[,j] = rnorm(n)
  X[,1] = hinge_r(X[,1],0)
  X[,2] = hinge_r(X[,1],1)
  y = X[,1] + X[,2]  + rnorm(n,sd=0.1)
}

if(sim=="4"){
  #Example 4:
  n = 1000
  p = 50
  X = matrix(0,ncol=p,nrow=n)
  for(j in 1:p) X[,j] = rnorm(n)
  y = sin(X[,1])  + rnorm(n,sd=0.1)
}
if(sim=="5"){
  #Example 5:
  n = 1000
  p = 50
  X = matrix(0,ncol=p,nrow=n)
  for(j in 2:p) X[,j] = rnorm(n)
  X[,1]=sort(rnorm(n))
  y = ifelse(X[,1]<0,1,0)
}

library(earth)
d=data.frame(X,y)
#Dopasowanie modelu:
model = earth(y~.,data=d,degree=2,trace=3,penalty=0,minspan=0)
#Rownanie modelu:
summary(model, digits = 2, style = "pmax")

#Wykresy Y~X oraz \hat{Y}~X
or1 = order(X[,1])
plot(X[or1,1],y[or1])
lines(X[or1,1],model$fitted.values[or1],type="l",col="red",lwd=2)

#Wykres: Y~\hat{Y}
plot(y[or1],model$fitted.values[or1])


###### ZADANIE 2 #######

# Do samodzielnego wykonania.
