# https://home.ipipan.waw.pl/p.teisseyre/TEACHING/ZMUM/Zadania/Bayesian.pdf
# Task 1

# LDA
my_lda = function(x, y) {
  x1 = x[y == 1, ]
  x0 = x[y == 0, ]
  
  pi1 = nrow(x1) / nrow(x)
  pi0 = nrow(x0) / nrow(x)
  
  mu1 = apply(x1, 2, mean)
  mu0 = apply(x0, 2, mean)
  
  sigma = cov(x)
  sigma_inv = solve(sigma)
  det1 = det(sigma)
  
  return (list(mu1 = mu1, mu0 = mu0, sigma_inv = sigma_inv, pi1 = pi1, pi0 = pi0, det1 = det1))
}

my_lda_predict = function(object, xtest) {
  mu1 <- object$mu1
  mu0 <- object$mu0
  sigma_inv <- object$sigma_inv
  det1 <- object$det1
  pi1 <- object$pi1
  pi2 <- object$pi2
  posterior <- matrix(0, nrow = nrow(xtest), ncol = 2)
  for(i in 1:nrow(xtest)) {
    x0 <- as.numeric(xtest[i, ])
    form1 <- as.numeric(t((x0 - mu1)) %*% sigma_inv %*% (x0 - mu1))
    form0 <- as.numeric(t((x0 - mu0)) %*% sigma_inv %*% (x0 - mu0))
    px1 <- (1 / ((2 * pi) ^ (p  / 2) * sqrt(det1))) * exp(-0.5 * form1)
    px0 <- (1 / ((2 * pi) ^ (p  / 2) * sqrt(det1))) * exp(-0.5 * form0)
    posterior[i, 1] <- px0 * pi0 / (px1 * pi1 + px0 * pi0)
    posterior[i, 2] <- px1 * pi1 / (px1 * pi1 + px0 * pi0)
  }
  return (posterior)
}

# QDA
my_qda = function(x, y) {
  x1 = x[y == 1, ]
  x0 = x[y == 0, ]
  
  pi1 = nrow(x1) / nrow(x)
  pi0 = nrow(x0) / nrow(x)
  
  mu1 = apply(x1, 2, mean)
  mu0 = apply(x0, 2, mean)
  
  sigma1 = cov(x1)
  sigma0 = cov(x0)
  sigma_inv1 = solve(sigma1)
  sigma_inv0 = solve(sigma0)
  det1 = det(sigma1)
  det0 = det(sigma0)
  
  return (list(mu1 = mu1, mu0 = mu0, sigma_inv1 = sigma_inv1, sigma_inv0 = sigma_inv0, pi1 = pi1, pi0 = pi0, det1 = det1, det0 = det0))
}

my_qda_predict = function(object, xtest) {
  mu1 <- object$mu1
  mu0 <- object$mu0
  sigma_inv1 <- object$sigma_inv1
  sigma_inv0 <- object$sigma_inv0
  det1 <- object$det1
  det0 <- object$det0
  pi1 <- object$pi1
  pi2 <- object$pi2
  posterior <- matrix(0, nrow = nrow(xtest), ncol = 2)
  for(i in 1:nrow(xtest)) {
    x0 <- as.numeric(xtest[i, ])
    form1 <- as.numeric(t((x0 - mu1)) %*% sigma_inv1 %*% (x0 - mu1))
    form0 <- as.numeric(t((x0 - mu0)) %*% sigma_inv0 %*% (x0 - mu0))
    px1 <- (1 / ((2 * pi) ^ (p  / 2) * sqrt(det1))) * exp(-0.5 * form1)
    px0 <- (1 / ((2 * pi) ^ (p  / 2) * sqrt(det0))) * exp(-0.5 * form0)
    posterior[i, 1] <- px0 * pi0 / (px1 * pi1 + px0 * pi0)
    posterior[i, 2] <- px1 * pi1 / (px1 * pi1 + px0 * pi0)
  }
  return (posterior)
}

library('ISLR')
Default$student <- ifelse(Default$student == 'Yes', 1, 0)
Default$default <- ifelse(Default$default == 'Yes', 1, 0)
train <- Default[1:8000, ]
test <- Default[8001:10000, ]
q <- my_qda(train[, 2:4], train[, 1])
my_qda_predict(q, test)


