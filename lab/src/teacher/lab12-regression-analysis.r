### ZAD1 ###
rel=read.table("realest.txt",header=T)
rel
attach(rel)

# (a)
l=lm(Price~.,data=rel)
summary(m)
# (b)

# (c)

predict(l,newdata=data.frame(Bedroom=3,Space=1500,Room=8,Lot=40,Bathroom=2,Garage=1,Tax=1000,Condition=0))
#76.21223

detach(rel)

### ZAD2 ###
library(car)
data(USPop)
attach(USPop)
plot(year, population,type="b")

#a)
time <- 0:21
pop.mod <- nls(population ~ beta1/(1 + exp(beta2 + beta3*time)),
               start=list(beta1 = 350,beta2 = 4.5, beta3 = -0.3),
               trace=T)

plot(year, population)
lines(year, fitted.values(pop.mod), lwd=2)

#b)
predict(pop.mod,list(time=22.5))

### ZAD3 ###
krzywa = function(x) {
  4.26*(exp(-x)-4*exp(-2*x)+3*exp(-3*x))
}

n=1000
x=runif(n,min=0,max=3.5)
x=sort(x)
e=rnorm(n,0,0.1)
y=krzywa(x)+e

yteor=krzywa(x)

plot(x,y)
lines(x,yteor)

#Estymator jadrowy:
est.ksmooth = ksmooth(x,y)

#Smooth Spline:
est.spline=smooth.spline(x,y)

#Lokalnie wielomianowy:
est.loess1=loess(y~x)


plot(x,y,main="Jadrowy")
lines(x,yteor,lwd=2)
lines(est.ksmooth,col="red",lwd=2)
lines(x,est.loess1$fitted,lty=1,col="blue")
lines(est.spline,col="orange",lwd=2)


#b)
ISE=function(x,y) mean((x-y)^2)

ISE(y,est.loess1$fitted)
ISE(y,est.ksmooth$y)
ISE(y,est.spline$y)

#c)
est.loess01=loess(y~x,span=0.1)
est.loess04=loess(y~x,span=0.4)
est.loess075=loess(y~x,span=0.75)

plot(x,y,main="loess")
lines(x,yteor)
lines(x,est.loess01$fitted,lty=1,col="blue")
lines(x,est.loess04$fitted,lty=1,col="green")
lines(x,est.loess075$fitted,lty=1,col="red")
legend("bottomright",c("span=0.1","sapn=0.4","span=0.75"),col=c("blue","green","red"),lty=c(1,1,1))

#d)
pred.loess=predict(est.loess04,data.frame(x=2),se=TRUE)
pred.loess$fit

krzywa(2)

abs(pred.loess$fit-krzywa(2))
