###############################################
######### ANALIZA SKLADOWYCH GLOWNYCH #########
###############################################


###### ZADANIE 1 #######

#Wczytanie danych:
data(USArrests)

#Standaryzacja danych:
USArrests = data.frame(scale(USArrests))

#Wyznaczenie macierzy kowariancji:
cov1 = cov(USArrests)

#Rozklad spektralny:
eigen1 = eigen(cov1)

#Wektory ladunkow (w kolumnach):
loadings = eigen1$vectors

#Skladowe glowne:
scores = as.matrix(USArrests) %*% loadings

#Odchylenia standardowe skladowych glownych:
sds = sqrt(eigen1$values)

#Analiza skladowych głownych przy pomocy funkcji princomp:
pca = princomp(~.,data=USArrests)

#Porownanie wynikow:
print(loadings)
print(pca$loadings)

print(scores[1:10,])
print(pca$scores[1:10,])


###### ZADANIE 2 #######

#Wczytanie danych:
data(USArrests)

#Standaryzacja danych:
USArrests = data.frame(scale(USArrests))

#Analiza skladowych glownych:
pca = princomp(~.,data=USArrests)

#Ile zmiennosci danych tlumacza poszczegolne skladowe:
vars = pca$sdev^2
vars = vars/sum(vars)

pdf("cumvar.pdf")
par(oma=c(0,0,0,0))
plot(1:4,cumsum(vars),type="b",col="orange",lwd=2,xlab="Principal component",ylab="Cumulative Proportion of Variance Explained",cex.lab=1.4)
dev.off()

#Interpretacja dwoch pierwszych skladowych:
pdf("biplot.pdf")
par(oma=c(0,0,0,0))
biplot(pca,col=c("black","orange"),cex=c(0.8,0.9),cex.lab=1.4)
dev.off()


###### ZADANIE 3 #######
library(ISLR)

#Wczytanie danych:
data(Hitters)

#Ze wzgledow technicznych wstawiamy zmienna Salary jako ostatnia kolumne:
Hitters=Hitters[,c(1:18,20,19)]

#Usuwamy wiersze z brakami danych:
Hitters=na.omit(Hitters)

#Wykonaujemy PCA na zmiennych objasniajacych:
x=model.matrix(Salary~.,Hitters)[,-1]
x = data.frame(scale(x))
pca = princomp(x)
plot(pca)

#Dopasowujemy model liniowy uzywajac skladowych glownych, sprawdzamy ktore skladowe sa najbardziej istotne w modelu.
dataTemp = data.frame(pca$scores,Hitters$Salary)
names(dataTemp)[ncol(dataTemp)]="Salary"
lm.fit1 = lm(Salary~.,data=dataTemp)
print(summary(lm.fit1))


#R2 vs skladowe glowne:
p = ncol(x)
pca.r2 = numeric(p)
for(j in 1:p){
  lm.fit = lm(Salary~.,data=dataTemp[,c(1:j,ncol(dataTemp))])
  pca.r2[j] = summary(lm.fit)$r.squared
}

plot(1:p,pca.r2,type="b",col="orange",lwd=2,xlab="Variables",ylab="R2")

#R2 vs najbardziej istotne zmienne z modelu liniowego:
p = ncol(x)
lm.r2 = numeric(p)
lm.fit0 = lm(Salary~.,data=Hitters)
tstat = abs(summary(lm.fit0)$coef[,3])[-1]
order1 = order(tstat,decreasing=T)

for(j in 1:p){
  lm.fit = lm(Salary~.,data=Hitters[,c(order1[1:j],ncol(Hitters))])
  lm.r2[j] = summary(lm.fit)$r.squared
}
#Wykres R2 vs najbardziej istotne skladowe/zmienne:
plot(1:p,pca.r2,type="b",col="orange",lwd=2,xlab="Variables",ylab="R2",ylim=c(0,0.6))
lines(1:p,lm.r2,col="blue",lwd=2,type="b")
legend("bottomright",c("PCA","LM"),col=c("orange","blue"),lwd=c(2,2),lty=c(1,1))







