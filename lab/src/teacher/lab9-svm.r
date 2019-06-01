###############################################
###### MASZYNY WEKTOROW PODPIERAJACYCH ########
###############################################


###### ZADANIE 1 #######
# generujemy dane wedlug opisu
y <- c(rep(1, 500), rep(0, 500))
x <- matrix(0, ncol = 2, nrow = 1000)

# generujemy obserwacje z klasy 1
x[1:500, 1] <- runif(500, -1, 1)
x[1:500, 2] <- sample(x = c(-1, 1), size = 500, replace = T, prob = c(0.5, 0.5))*sqrt(1-x[1:500, 1]^2)
# generujemy obserwacje z klasy 0
x[501:1000, 1] <- runif(500, -2, 2)
x[501:1000, 2] <- sample(x = c(-1, 1), size = 500, replace = T, prob = c(0.5, 0.5))*sqrt(4-x[501:1000, 1]^2)

# dodajemy szum do zmiennej x2
x[, 2] <- x[, 2] + rnorm(1000, 0, 0.1)

# wykres rozproszenia zmiennych w klasach
plot(x[, 1], x[, 2], col = as.factor(y))

# lub w ggplot2
library(ggplot2)
q <- ggplot(data = data.frame(y = y, x = x), aes(x = x.1, y = x.2, col = as.factor(y)))+
  geom_point(size = 2.5) + 
  xlab("x1") + ylab("x2") + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2)) + 
  labs(colour = "y")
q

ggsave(filename = "svm.pdf", plot = q, width = 10, height = 8)



# dla tak wygenerowanych danych testujemy dzialanie metody svm z roznymi jadrami

library(e1071)

model1 <- svm(x, as.factor(y), kernel = "linear")
yhat1 <- predict(model1, x)

q1 <- ggplot(data = data.frame(y = yhat1, x = x), aes(x = x.1, y = x.2, col = as.factor(y)))+
  geom_point(size = 2.5) + 
  xlab("x1") + ylab("x2") + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2)) + 
  labs(colour = "y")
q1


################################################################################
model2 <- svm(x, as.factor(y), kernel = "radial")
yhat2 <- predict(model2, x)

q2 <- ggplot(data = data.frame(y = yhat2, x = x), aes(x = x.1, y = x.2, col = as.factor(y)))+
  geom_point(size = 2.5) + 
  xlab("x1") + ylab("x2") + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2)) + 
  labs(colour = "y")
q2




################################################################################
model3 <- svm(x, as.factor(y), kernel = "polynomial", degree = 5)
yhat3 <- predict(model3, x)

q3 <- ggplot(data = data.frame(y = yhat3, x = x), aes(x = x.1, y = x.2, col = as.factor(y)))+
  geom_point(size = 2.5) + 
  xlab("x1") + ylab("x2") + 
  theme(axis.title.x = element_text(size = 15, angle = 0, face = "italic"), 
        axis.title.y = element_text(size = 15, angle = 90, face = "italic")) +
  theme(legend.title = element_text(size = 14, face = "bold")) +
  theme(legend.text = element_text(colour = "black", size = 12)) + 
  guides(color = guide_legend(keywidth = 2, keyheight = 2)) + 
  labs(colour = "y")
q3
