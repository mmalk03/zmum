setwd("folder z danymi")

###################################################################
###############            ZADANIE 1           ####################
###################################################################

# generujemy 200 elementowa probe z zadanego rozkladu
set.seed(7777)
n <- 200
w <- rbinom(n, 1, p)
x <- w * rnorm(n, 5, sd = 1) + (1-w) * rnorm(n, 10, sd = 1)

# wyznaczenie teoretcznej wartosci gestosci w wylosowanych punktach
x1 <- sort(x)
theoreticalDensity <- p * dnorm(x1, 5, 1) + (1-p) * dnorm(x1, 10, 1)

# histogram z naniesiona teoretyczna krzywa gestosci
hist(x1, freq = F, ylim = c(0, max(theoreticalDensity)), breaks = 15)
points(x1, theoreticalDensity, type = "l")

# a) - wyznaczamy estymator jadrowy gestosci z jadrem gaussowskim
empiricalDensity <- density(x, kernel = "gaussian")

# na jeden wykres nanosimy histogram (czarna linia), wyznaczony estymator gestosci 
# (czerwona linia) oraz wykres getosci podanej mieszanki (niebieska linia)
hist(x, breaks = 15, probability = TRUE)
points(x1, theoreticalDensity, col = "blue", type = "l")
lines(empiricalDensity, type = "l", col = "red")


# b) - wyznaczamy empiryczna wartosc bledu sredniokwadratowego

empiricalDensity2 <- density(x, kernel = "gaussian", from = 2, to = 12, n = 512)
x2 <- empiricalDensity2$x

f1 <- p * dnorm(x2, 5, 1) + (1-p) * dnorm(x2, 10, 1)

# wartosc bledu sredniokwadratowego
mean((f1-empiricalDensity2$y)^2)


###################################################################
###############            ZADANIE 2           ####################
###################################################################

library(MASS)

data(geyser)

x <- geyser$waiting
y <- geyser$duration

# wyznaczamy rozstepy metoda Shealtera-Jonesa dla kazdej zmiennej oddzielnie
h1 <- bw.SJ(x)
h2 <- bw.SJ(y)

# wyznaczam estymator gestosci
f.SJ <- kde2d(x, y, n = 100, h = c(h1, h2))
persp(f.SJ, col = "orange", phi = 10, theta = -50, xlab = "Waiting", ylab = "Duration", zlab = "Denstiy")

# co sie stanie gdy zwieksze rozstepy??
h11 <- 20*h1
h22 <- 20*h2
f.SJ <- kde2d(x, y, n = 100, h = c(h11, h22))
persp(f.SJ, col = "orange", phi = 10, theta = -50, xlab = "Waiting", ylab = "Duration", zlab = "Denstiy")


###################################################################
###############            ZADANIE 3           ####################
###################################################################

A <- read.table("https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/DANE/earthquake.txt")

attach(A)

# a) - wyznaczamy gestosci dla zmiennej body w obu populacjach
den_equake <- density(body[popn == "equake"], bw = 0.2)
den_explosn <- density(body[popn == "explosn"], bw = 0.2)

plot(den_equake, type = "l", col = "red", ylim = c(0, 2))
lines(den_explosn, col = "blue")
legend("topleft", c("equake", "explosion"), col = c("red", "blue"), lty = c(1, 1))


# ggplot
library(ggplot2)

den_data <- data.frame(x = c(den_explosn$x, den_equake$x), 
                       y = c(den_explosn$y, den_equake$y), 
                       popn = c(rep("explosn", length(den_explosn$x)), 
                                rep("equake", length(den_equake$x)))) 

ggplot(data = den_data, aes(x = x, y = y, col = popn)) + 
  geom_line() + xlab("N = 20  Bandwidth = 0.2") + ylab("Density") + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))


# wyznaczamy p-stwa apriori
p_equake <- sum(popn == "equake")/length(popn)
p_explosn <- sum(popn == "explosn")/length(popn)

plot(den_equake$x, p_equake*den_equake$y, type = "l", col = "red", ylim = c(0, 1))
lines(den_explosn$x, p_explosn*den_explosn$y, col = "blue")
legend("topleft", c("equake", "explosion"), col = c("red", "blue"), lty = c(1, 1))


den_data2 <- data.frame(x = c(den_explosn$x, den_equake$x), 
                        y = c(p_explosn*den_explosn$y, p_equake*den_equake$y), 
                        popn = c(rep("explosn", length(den_explosn$x)), 
                                 rep("equake", length(den_equake$x)))) 

ggplot(data = den_data2, aes(x = x, y = y, col = popn)) + 
  geom_line() + xlab("x") + ylab("y") + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))



# b) - wyznaczamy estymatory gestosci dwuwymiarowych w obu populacjach
library(MASS)
f_equake <- kde2d(body[popn == "equake"], surface[popn == "equake"])
f_explosn <- kde2d(body[popn == "explosn"], surface[popn == "explosn"])

# konturowe wykresy gestosci
contour(f_equake, col = "red")
contour(f_explosn, add = T, col = "blue")
points(body[popn == "equake"], surface[popn == "equake"], pch = 19, col = "red")
points(body[popn == "explosn"], surface[popn == "explosn"], pch = 19, col = "blue")

# ggplot
den_data_equake <- cbind(expand.grid(f_equake$x, f_equake$y), z = as.numeric(f_equake$z), popn = "equake")
den_data_explosn <- cbind(expand.grid(f_explosn$x, f_explosn$y), z = as.numeric(f_explosn$z), popn = "explosn")
colnames(den_data_equake) <- colnames(den_data_explosn) <- c("body", "surface", "density", "popn")


ggplot() + 
  geom_contour(data = den_data_equake, 
               aes(x = body, y = surface, z = density, col = popn), col = "red", 
               linetype = 2, size = 1, alpha = 0.5) + 
  geom_contour(data = den_data_explosn, 
               aes(x = body, y = surface, z = density, col = popn), col = "blue", 
               linetype = 2, size = 1, alpha = 0.5) + 
  geom_point(data = A, aes(x = body, y = surface, col = popn), size = 2.5) + 
  scale_color_manual(values = c("red", "blue")) + 
  xlab("body") + ylab("surface") + ggtitle("2-dim density") + 
  theme(plot.title = element_text(lineheight  = .8, face = "bold", hjust = 0.5)) + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2))


# trojwymiarowe wykresy gestosci
persp(f_equake, col = "red")
persp(f_explosn, col = "blue")


###################################################################
###############            ZADANIE 4           ####################
###################################################################

# Do samodzielngo wykonania.