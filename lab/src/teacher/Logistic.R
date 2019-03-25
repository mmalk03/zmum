###############################################
############ REGRESJA LOGISTYCZNA #############
###############################################


###### ZADANIE 1 #######

SA <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/SAheart.data", sep = ",", h = T)
# a)

# X = 1 (chd = 1), X = 0 (chd = 0)
# Y = 1 (famhist = present), Y = 0 (famhist = absent) 

#      X = 1     X = 0
# Y=1  p11        p10
# Y=0  p01        p00

# OR = (p11p00)/(p01p10)
n <- nrow(SA)

n11 <- sum(SA$chd == 1 & SA$famhist == "Present")
n10 <- sum(SA$chd == 0 & SA$famhist == "Present")
n01 <- sum(SA$chd == 1 & SA$famhist == "Absent") 
n00 <- sum(SA$chd == 0 & SA$famhist == "Absent") 

# p11 = n11/n
# p10 = n10/n
# p01 = n01/n
# p00 = n00/n

# wyznaczamy iloraz szans dla zmiennych famhist i chd
OR <- (n11*n00)/(n01*n10)
OR


# b)  
SA <- SA[,-1]
SA.logit <- glm(chd ~ ., data = SA, family = "binomial")

# c)
summary(SA.logit)

# d) 
exp(as.numeric(coef(SA.logit)["age"]))

# e)
SA.logit.aic <- step(SA.logit, direction = "backward", k = 2) # selekcja z uzyciem kryterium AIC
summary(SA.logit.aic)
SA.logit.bic <- step(SA.logit, direction = "backward", k = log(nrow(SA))) # selekcja z uzyciem kryterium BIC
summary(SA.logit.bic)



###### ZADANIE 2 #######
library(ggplot2)

dane <- read.table("http://www.ipipan.eu/~teisseyrep/TEACHING/ZMUM/DANE/earthquake.txt", h = T)

#Wykres: 
ggplot(data = dane, aes(x = body, y = surface, col = popn))+
  geom_point(size = 2.5) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))
# z powyzszego wykresu widac, ze klasy w tych danych sa liniowo separowalne

model <- glm(popn ~ ., data = dane, family = "binomial")

# W przypadku liniowo separowalnych klas, algorytm Newtona nie zbiega!

summary(model)
# Otrzymujemy kuriozalny wynik (zmienne sa nieistotne w modelu). To jest zwiazane z ogromnymi bledami estymatora.



###### ZADANIE 3 #######
L = 50 #liczba symulacji
N = seq(from=50,to=300,by=10)

MSE = numeric(length(N))

i=1
for(n in N){
  cat("Simulation for n=",n,"\n")  
  mse = numeric(L)  
  for(k in 1:L){
    x1 = rnorm(n,0,1)
    x2 = rnorm(n,0,1)
    Beta = c(0.5,1,1)
    z =  Beta[1] + Beta[2]*x1 +Beta[3]*x2 
    p = 1/(1+exp(-z))
    y = rbinom(n,1,p)
    model = glm(y~x1+x2,family="binomial")
    Beta_Hat = as.numeric(model$coef)
    mse[k] =   sum((Beta_Hat-Beta)^2)
  }
  
  MSE[i] = mean(mse)
  i=i+1
}
plot(N,MSE,type="b",col="red")


