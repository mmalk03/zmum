###############################################
######### KLASYFIKACJA BAYESOWSKA #############
###############################################


###### ZADANIE 1 #######

#Wlasna implementacja metody LDA:
myLDA = function(x,y){
  x1 = x[y==1,]
  x0 = x[y==0,]
  
  pi1 = nrow(x1)/nrow(x)
  pi0 = nrow(x0)/nrow(x)
  
  mu1 = apply(x1,2,mean)
  mu0 = apply(x0,2,mean)
  Sigma = cov(x) 
  Sigma_inv = solve(Sigma)
  det1 = det(Sigma)
  return(list(mu1=mu1,mu0=mu0,Sigma_inv=Sigma_inv,pi1=pi1,pi0=pi0,det1=det1))
}
predict_myLDA = function(object,xtest){
  
  p = ncol(xtest)
  pi1 = object$pi1
  pi0 = object$pi0
  mu1 = object$mu1
  mu0 = object$mu0
  Sigma_inv = object$Sigma_inv
  det1 = object$det1
  posterior = matrix(0,nrow=nrow(xtest),ncol=2)
  for(i in 1:nrow(xtest)){
    x0 = as.numeric(xtest[i,])
    form1 = as.numeric(t((x0-mu1))%*%Sigma_inv%*%(x0-mu1))
    form0 = as.numeric(t((x0-mu0))%*%Sigma_inv%*%(x0-mu0))
    px1 = (1/((2*pi)^(p/2)*sqrt(det1)))*exp(-0.5*form1)
    px0 = (1/((2*pi)^(p/2)*sqrt(det1)))*exp(-0.5*form0)
    posterior[i,1] = px0*pi0/(px1*pi1+px0*pi0)
    posterior[i,2] = px1*pi1/(px1*pi1+px0*pi0)
  }
  return(posterior)
}

#Wczytanie danych:
library(ISLR)
data(Default)
Default$student=ifelse(Default$student=="Yes",1,0)
Default$default=ifelse(Default$default=="Yes",1,0)

x = Default[,2:4]
y = Default[,1]

#Wywolanie funkcji myLDA i predict_myLDA:
myLDA1 = myLDA(x,y)
posterior1 = predict_myLDA(myLDA1,xtest)

#Porownanie z funkcja lda z pakietu MASS:
library(MASS)
lda1 = lda(default~.,data=Default)
posterior2 = predict(lda1,x)$posterior

#Wyswieltamy numery obserwacji z najwiekszymi pstwami aposteriori dla myLDA i lda:
print(order(posterior1[,2])[1:10])
print(order(posterior2[,2])[1:10])
plot(posterior1[,2],posterior2[,2])


###### ZADANIE 2 #######

n=1000
p=2

#Schemat 1:
x = matrix(0,nrow=n,ncol=p)
y = rbinom(n,1,0.5)
w1 = which(y==1)
w0 = which(y==0)
for(j in 1:p){
  x[w1,j] = rnorm(length(w1),1,1)
}
for(j in 1:p){
  x[w0,j] = rnorm(length(w0),0,1)
}

data = data.frame(x,y)

model1 = lda(y~.,data=data)
pred1 = predict(model1,data,type=) 
# Macierz klasyfikacji:
tab1 = table(y,pred1$class)
print(tab1)
cat("Accuracy for LDA =",sum(diag(tab1))/sum(tab1))

model2 = qda(y~.,data=data)
pred2 = predict(model2,data,type=) 
# Macierz klasyfikacji:
tab2 = table(y,pred2$class)
print(tab2)
cat("Accuracy for QDA =",sum(diag(tab2))/sum(tab2))


#Schemat 2:
# rho = 0.8
# var1  = (1-rho^2)/(rho^2)
# sd1 = sqrt(var1)

x = matrix(0,nrow=n,ncol=p)
y = rbinom(n,1,0.5)
w1 = which(y==1)
w0 = which(y==0)

Sigma0 = matrix(c(1,0.8,0.8,1),ncol=2,nrow=2)
x[w0,] = mvrnorm(n = length(w0), mu=c(0,0), Sigma=Sigma0)

Sigma1 = matrix(c(1,-0.8,-0.8,1),ncol=2,nrow=2)
x[w1,] = mvrnorm(n = length(w1), mu=c(1,1), Sigma=Sigma1)


data = data.frame(x,y)

model1 = lda(y~.,data=data)
pred1 = predict(model1,data,type=) 
# Macierz klasyfikacji:
tab1 = table(y,pred1$class)
print(tab1)
cat("Accuracy for LDA =",sum(diag(tab1))/sum(tab1))

model2 = qda(y~.,data=data)
pred2 = predict(model2,data,type=) 
# Macierz klasyfikacji:
tab2 = table(y,pred2$class)
print(tab2)
cat("Accuracy for QDA =",sum(diag(tab2))/sum(tab2))



###### ZADANIE 3 #######

#Wczytanie danych:
Wina <- read.table("http://www.ipipan.eu/~teisseyrep/TEACHING/DM/DANE/wine.data", sep = ",")
Wina$V1 <- as.factor(Wina$V1)

library(MASS)
library(ggplot2)

w <- sample(1:nrow(Wina), size = ceiling(2/3*nrow(Wina)))
train <- Wina[w,]
test <- Wina[-w,]

#Dopasowanie modeli na danych treningowych:

# LDA
train.lda <- lda(V1 ~ V2 + V8, data = train)

# QDA
train.qda <- qda(V1 ~ V2 + V8, data = train)



#Graficzna reprezentacja klasyfikatora:

ile <- 120 # parametr okreslajacy gestosc siatki punktow
xp <- seq(11, 15, length = ile)
yp <- seq(0.5, 5, length = ile)
siatka <- expand.grid(V2 = xp, V8 = yp)


# dla LDA
z11 <- predict(train.lda, newdata = siatka)
head(z11$posterior)
# zp1 - pstwo aposteriori klasy 1 - max(pstwo aposteriori z klas 2 i 3)
# zp1 > 0 - klasyfikuj do grupy 1.
zp1 <- z11$post[, 1] - pmax(z11$post[, 2], z11$post[, 3])
zp2 <- z11$post[, 2] - pmax(z11$post[, 1], z11$post[, 3])
zp3 <- z11$post[, 3] - pmax(z11$post[, 1], z11$post[, 2])

plot(Wina[, c(2, 8)], type = "n", xlab = "Alcohol", ylab = "Flavanoids", 
     main = "LDA Classification")
text(Wina$V2, Wina$V8, as.character(Wina$V1), cex = 0.8)
# dorysujemy poziomice odpowiadajaca wartosci 0 dla zmiennych zp1, zp2, zp3 - sa 
# to 'granice' przynaleznosci klasowej wyznaczone na podstawie metody LDA
contour(xp, yp, matrix(zp1, ile), level = 0, add = T, col = "red")
contour(xp, yp, matrix(zp2, ile), level = 0, add = T, col = "green")
contour(xp, yp, matrix(zp3, ile), level = 0, add = T, col = "blue")


# alternatywnie przy pomocy ggplot2
q <- ggplot() + 
  geom_contour(data = cbind(siatka, data.frame(pred2 = zp2)), 
               aes(x = V2, y = V8, z = pred2), bins = 1, col = "green", 
               linetype = 1, size = 1.2, alpha = 0.5) + 
  geom_contour(data = cbind(siatka, data.frame(pred3 = zp3)), 
               aes(x = V2, y = V8, z = pred3), bins = 1, col = "blue", 
               linetype = 2, size = 1.2) + 
  geom_contour(data = cbind(siatka, data.frame(pred1 = zp1)), 
               aes(x = V2, y = V8, z = pred1), bins = 1, col = "red", 
               linetype = 4, size = 1.2) + 
  geom_point(data = Wina, aes(x = V2, y = V8, col = V1), size = 2) + 
  xlab("Alcohol") + ylab("Flavanoids") + ggtitle("LDA Classification") + 
  theme(plot.title = element_text(lineheight  = .8, face = "bold", hjust = 0.5)) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))
q

# ggsave(plot = q, filename = "LDA.pdf", width = 7.5, height = 5)


# dla QDA
z11 <- predict(train.qda, newdata = siatka)
head(z11$posterior)
#zp1 - pstwo aposteriori klasy 1 - max(pstwo aposteriori z klas 2 i 3) 
#zp1 > 0 - klasyfikuj do grupy 1.
zp1 <- z11$post[, 1] - pmax(z11$post[, 2], z11$post[, 3])
zp2 <- z11$post[, 2] - pmax(z11$post[, 1], z11$post[, 3])
zp3 <- z11$post[, 3] - pmax(z11$post[, 1], z11$post[, 2])

plot(Wina[, c(2, 8)], type = "n", xlab = "Alcohol", ylab = "Flavanoids", main = "LDA Classification")
text(Wina$V2, Wina$V8, as.character(Wina$V1), cex = 0.8)
# dorysujemy poziomice odpowiadajaca wartosci 0 dla zmiennych zp1, zp2, zp3 - sa 
# to 'granice' przynaleznosci klasowej wyznaczone na podstawie metody QDA
contour(xp, yp, matrix(zp1, ile), level = 0, add = T, col = "red")
contour(xp, yp, matrix(zp2, ile), level = 0, add = T, col = "green")
contour(xp, yp, matrix(zp3, ile), level = 0, add = T, col = "blue")


# alternatywnie przy pomocy ggplot2
q <- ggplot() + 
  geom_contour(data = cbind(siatka, data.frame(pred2 = zp2)), 
               aes(x = V2, y = V8, z = pred2), bins = 1, col = "green", 
               linetype = 1, size = 1.2, alpha = 0.5) + 
  geom_contour(data = cbind(siatka, data.frame(pred3 = zp3)), 
               aes(x = V2, y = V8, z = pred3), bins = 1, col = "blue", 
               linetype = 2, size = 1.2) + 
  geom_contour(data = cbind(siatka, data.frame(pred1 = zp1)), 
               aes(x = V2, y = V8, z = pred1), bins = 1, col = "red", 
               linetype = 4, size = 1.2) + 
  geom_point(data = Wina, aes(x = V2, y = V8, col = V1), size = 2) + 
  xlab("Alcohol") + ylab("Flavanoids") + ggtitle("QDA Classification") + 
  theme(plot.title = element_text(lineheight  = .8, face = "bold", hjust = 0.5)) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))
q

###### ZADANIE 4 #######
library(klaR)

#Wczytanie danych:
Kredit <- read.table("http://www.ipipan.eu/~teisseyrep/TEACHING/DM/DANE/kredit.asc", h = T)


#Korkowa eliminacja zmiennych oparta o minimalizację bledu klasyfikacji:
step1 <- stepclass(kredit ~ ., data = Kredit, method = "lda", direction = "backward")
plot(step1)


