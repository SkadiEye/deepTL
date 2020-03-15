library(deepTL)
library(MASS)
n <- 10000
p <- 5
n_test <- 10000

par(mfrow = 2:3)
n_ensemble <- 10
esCtrl1 <- list(n.hidden = c(50, 40, 30, 20),
                l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam")

#### zinb model ####
n <- 1000
theta <- 3
esCtrl1$n.batch <- 100
esCtrl1$n.epoch <- 100

x <- matrix(runif(n*p), n, p)
z <- rbinom(n, 1, 1/(1 + exp(rowSums(x)*2 - p)))
w <- rnegbin(n, exp(x[, 1]**2 + x[, 2]**2)*2, theta)
y <- z*w

x_test <- matrix(runif(n_test*p), n_test, p)
z_test <- rbinom(n_test, 1, 1/(1 + exp(rowSums(x_test)*2 - p)))
w_test <- rnegbin(n_test, exp(x_test[, 1]**2 + x_test[, 2]**2)*2, theta)
y_test <- z_test*w_test

dnn_dat <- importDnnet(x, y)

zinb_mod <- ensemble_dnnet(dnn_dat, n_ensemble, esCtrl = appendArg(esCtrl1, "family", "zinb", TRUE))
zinb_pred <- predict(zinb_mod, x_test)

zip_mod <- ensemble_dnnet(dnn_dat, n_ensemble, esCtrl = appendArg(esCtrl1, "family", "zip", TRUE))
zip_pred <- predict(zip_mod, x_test)

head(cbind(1/(1 + exp(rowSums(x_test)*2 - p)),
           exp(x_test[, 1]**2 + x_test[, 2]**2)*2,
           zinb_pred, zip_pred))
head(cbind(1/(1 + exp(rowSums(x_test)*2 - p))*exp(x_test[, 1]**2 + x_test[, 2]**2)*2,
           apply(zinb_pred, 1, prod), apply(zip_pred, 1, prod)))

par(mfrow = c(2, 2))
plot(1/(1 + exp(rowSums(x_test)*2 - p)), zinb_pred[, 1], pch = 3,
     xlab = "True prob.", ylab = "Pred. prob.", main = "ZINB-DNN")
abline(0, 1, col = 2)
plot(1/(1 + exp(rowSums(x_test)*2 - p)), zip_pred[, 1], pch = 3,
     xlab = "True prob.", ylab = "Pred. prob.", main = "ZIP-DNN")
abline(0, 1, col = 2)
plot(x_test[, 1]**2 + x_test[, 2]**2 + log(2), log(zinb_pred[, 2]), pch = 3,
     xlab = "True log(lambda)", ylab = "Pred. log(lambda)", main = "ZINB-DNN")
abline(0, 1, col = 2)
plot(x_test[, 1]**2 + x_test[, 2]**2 + log(2), log(zip_pred[, 2]), pch = 3,
     xlab = "True log(lambda)", ylab = "Pred. log(lambda)", main = "ZIP-DNN")
abline(0, 1, col = 2)

zinb_mod@model.spec$negbin_alpha

