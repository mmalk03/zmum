###################################################################
###############            ZADANIE 1           ####################
###################################################################

library("rpart")
library("rpart.plot")
SA <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/SAheart.data", h = T, row.names = 1, sep = ",")

# a) - dopasujemy drzewo klasyfikacyjne z parametrami cp = 0.01 i minsplit = 5

# wazne, aby ustawic as.factor(chd) by zbudowac drzewo klasyfikacyjne a nie regresyjne
tree <- rpart(as.factor(chd) ~ ., data = SA, cp = 0.001, minsplit = 5)
pdf(file="tree.pdf")
rpart.plot(tree)
dev.off()

print(summary(tree), digits = 4)
# Opis drzewa:
#1 - numer wezla
#2 - Nazwa zmiennej wg, ktorej dokonano podzialu i warunek realizacji podzialu
#3 - Liczba elementow w wezle
#4 - Liczba elementow blednie zaklasyfikowanych
#5 - Predykcja przynaleznosci klasowej
#6 - wektor estymatorow pst. przynaleznosci klasowej 
#    (nazwy uporzadkowane leksograficznie)

# b) - graficzna reprezentacja drzewa
par(mar = c(0, 1, 0, 1))
plot(tree)
text(tree, use.n = TRUE)
par(mar = c(4, 4, 4, 4))

# c) - dokonujemy predykcji klasy dla obserwacji ze zmiennymi objasniajacymi bedacymi srednimi 
#      wartosciami zmiennych ze zbioru SA
x0 <- as.data.frame(t(apply(SA[,-c(5)], 2, mean)))
table(SA$famhist)
# Absent Present 
# 270     192 
x0 <- cbind(x0, famhist = "Absent")
predict(tree, newdata = rbind(x0), type = "class")


# d) - wybierzmy teraz drzewo optymalne w oparciu o regule 1SE
plotcp(tree)
# rysunek wykonany na podstawie tabeli cptable
tree$cptable

# przycinamy drzewo wybierajac parametr cp zgodnie z regula 1SE
Z <- prune.rpart(tree, cp = 0.04)
par(mar = c(0, 1, 0, 1))
plot(Z)
text(Z, use.n = TRUE)
par(mar = c(4, 4, 4, 4))



# wybor innego kryterium podzialu (miary roznorodnosci) mozna dokonac poprzez 
# ustawienie listy parametrow
tree.info <- rpart(as.factor(chd) ~ ., data = SA, cp = 0.01, minsplit = 5,
                   parms = list(split = "information"))
par(mar = c(0, 1, 0, 1))
plot(tree.info)
text(tree.info, use.n = TRUE)
par(mar = c(4, 4, 4, 4))

par(mar = c(0, 0, 0, 0))
par(mfrow = c(1, 2))
plot(tree)
text(tree, use.n = TRUE)
plot(tree.info)
text(tree.info, use.n = TRUE)
par(mfrow = c(1, 1))
par(mar = c(4, 4, 4, 4))


###################################################################
###############            ZADANIE 2           ####################
###################################################################

library(ggplot2)
Dane <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/earthquake.txt", h = T)

# a) - tworzymy wykres rozproszenia
q <- ggplot(data = Dane, aes(x = body, y = surface, col = popn))+
  geom_point(size = 2.5) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))
q

# b) - graficzna reprezentacja drzewa
# DRZEWO minsplit = 5:

Dane.tree <- rpart(as.factor(popn) ~ ., data = Dane, minsplit = 5) # dopasowujemy drzewo



par(mar = c(0, 0, 0, 0))
plot(Dane.tree)
text(Dane.tree)
par(mar = c(4, 4, 4, 4))

# tworzymy pomocnicze wartosci dla zmiennych objasniajacych
xp <- seq(4.5, 6.5, length = 50)
yp <- seq(3.5, 6.5, length = 50)
siatka <- expand.grid(body = xp, surface = yp)

# dokonujemy predykcji na sztucznie stworzonych obserwacjach ze zbioru siatka
P1 <- predict(Dane.tree, newdata = siatka, type = "prob")
zp1 <- ifelse(P1[,1] == 1, 1, -1)

# tworzymy wykres konturowy rozdzielajacy klasy wedlug reguly opartej o drzewo klasyfikacyjne
n1 <- 20
n2 <- 9
kol <- c(rep("black", n1), rep("red", n2))
sym <- c(rep("Q", n1), rep("X", n2))
plot(Dane$body, Dane$surface, type = "n", xlab = "body", ylab = "surface")
text(Dane$body, Dane$surface, sym, col = kol)
contour(xp, yp, matrix(zp1, 50), level = 0, add = T)


# alternatywnie przy pomocy ggplot2
q <- ggplot() + xlim(c(4.4, 6.6)) + ylim(c(3.4, 6.8)) + 
  geom_contour(data = cbind(siatka, data.frame(pred2 = zp1)), 
               aes(x = body, y = surface, z = pred2), bins = 1, col = "black", 
               linetype = 1, size = 1.2, alpha = 0.5) + 
  geom_point(data = Dane, aes(x = body, y = surface, col = popn), size = 2) + 
  xlab("body") + ylab("surface") + 
  theme(plot.title = element_text(lineheight  = .8, face = "bold", hjust = 0.5)) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))
q



# DRZEWO minsplit = 15:
Dane.tree <- rpart(as.factor(popn) ~ ., data = Dane, minsplit = 15)

par(mar = c(0, 0, 0, 0))
plot(Dane.tree)
text(Dane.tree)
par(mar = c(4, 4, 4, 4))

P2 <- predict(Dane.tree, newdata = siatka, type = "prob")
zp2 <- ifelse(P2[, 1] == 1, 1, -1)

n1 <- 20
n2 <- 9
kol <- c(rep("black", n1), rep("red", n2))
sym <- c(rep("Q", n1), rep("X", n2))
plot(Dane$body, Dane$surface, type = "n", xlab = "body", ylab = "surface")
text(Dane$body, Dane$surface, sym, col = kol)
contour(xp, yp, matrix(zp2, 50), level = 0, add = T)


# alternatywnie przy pomocy ggplot2
q <- ggplot() + xlim(c(4.5, 6.6)) + ylim(c(3.5, 6.5)) + 
  geom_contour(data = cbind(siatka, data.frame(pred2 = zp2)), 
               aes(x = body, y = surface, z = pred2), bins = 1, col = "black", 
               linetype = 1, size = 1.2, alpha = 0.5) + 
  geom_point(data = Dane, aes(x = body, y = surface, col = popn), size = 2) + 
  xlab("body") + ylab("surface") + 
  theme(plot.title = element_text(lineheight  = .8, face = "bold", hjust = 0.5)) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))
q



###################################################################
###############            ZADANIE 3           ####################
###################################################################

library(rpart)

Ft <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/fitness.txt", h = T)

# a) - dopasujemy drzewo regresyjne na podstawie wszystkich atrybutow
set.seed(123456)
Ft.tree <- rpart(Oxygen ~ ., data = Ft, cp = 0.01, minsplit = 2)
pdf(file="tree1.pdf")
rpart.plot(Ft.tree)
dev.off()
# graficzna reprezentacja drzewa
par(mar = c(0, 0, 0, 0))
plot(Ft.tree)
text(Ft.tree)
par(mar = c(4, 4, 4, 4))
# b) - dla jakiego biegacza pobor tlenu uznajemy za najwiekszy?
# ??????????????????????

# c) - dokonaj prognozy poboru tlenu dla obserwacji, ktorej wartosci 
#      zmiennych objasniajacych sa rowne medianom ze zbioru Ft
x0 <- apply(Ft[,-3], 2, median)
print(x0)

predict(Ft.tree, data.frame(t(x0)))


# d) - przytnijmy drzewo stosujac regule kosztu-zlozonosci oraz regule 1SE
plotcp(Ft.tree)
printcp(Ft.tree)

# regula kosztu-zlozonosci
Ft.tree.prune <- prune.rpart(Ft.tree, cp = 0.048)
par(mar = c(0, 0, 0, 0))
plot(Ft.tree.prune)
text(Ft.tree.prune)
par(mar = c(4, 4, 4, 4))

# regula 1SE
Ft.tree.prune <- prune.rpart(Ft.tree, cp = 0.14)
par(mar = c(0, 0, 0, 0))
plot(Ft.tree.prune)
text(Ft.tree.prune)
par(mar = c(4, 4, 4, 4))



# e) - budujemy drzewo na podstawie zmiennych RunTime i Age
Ft.tree1 <- rpart(Oxygen ~ ., data = Ft[,c(1, 3, 4)], cp = 0.02, minsplit = 2)

n_point <- 100
Age1 <- seq(from = 35, to = 60, length.out = n_point)
RunTime1 <- seq(from = 8, to = 15, length.out = n_point)

newdata <- expand.grid(Age = Age1, RunTime = RunTime1)
newdata$Z <- predict(Ft.tree1, newdata = newdata)

persp(Age1, RunTime1, matrix(newdata$Z, n_point), theta = 50, phi = 30, expand = 0.5, col = "lightblue", 
      zlab = "Oxygen prediction")

