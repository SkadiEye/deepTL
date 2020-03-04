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

#### zip model ####
x <- matrix(rnorm(n*p), n, p)
z <- rbinom(n, 1, 1/(1 + exp(x[, 3]**2 - x[, 4]**2 + x[, 5])))
w <- rpois(n, x[, 1]**2 + x[, 2]**2)
y <- z*w

x_test <- matrix(rnorm(n_test*p), n_test, p)
z_test <- rbinom(n_test, 1, 1/(1 + exp(x_test[, 3]**2 - x_test[, 4]**2 + x_test[, 5])))
w_test <- rpois(n_test, x_test[, 1]**2 + x_test[, 2]**2)
y_test <- z_test*w_test

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
args_dnnet <- appendArg(appendArg(esCtrl1, "train", dnn_spl$train, TRUE), "validate", dnn_spl$valid, TRUE)
args_dnnet <- appendArg(args_dnnet, "family", "zip", TRUE)
dnn_mod_cont <- do.call(dnnet, args_dnnet)
pred_cont <- predict(dnn_mod_cont, x_test)

bag_mod_cont <- ensemble_dnnet(dnn_dat, n_ensemble, esCtrl = appendArg(esCtrl1, "family", "zip", TRUE))
pred_cont_bag <- predict(bag_mod_cont, x_test)

head(cbind(1/(1 + exp(x_test[, 3]**2 - x_test[, 4]**2 + x_test[, 5])),
           x_test[, 1]**2 + x_test[, 2]**2,
           pred_cont, pred_cont_bag))
head(cbind(1/(1 + exp(x_test[, 3]**2 - x_test[, 4]**2 + x_test[, 5]))*(x_test[, 1]**2 + x_test[, 2]**2),
           apply(pred_cont, 1, prod), apply(pred_cont_bag, 1, prod)))

par(mfrow = c(2, 2))
plot(1/(1 + exp(x_test[, 3]**2 - x_test[, 4]**2 + x_test[, 5])), pred_cont[, 1], pch = 3,
     xlab = "True prob.", ylab = "Pred. prob.", main = "Single-DNN")
abline(0, 1, col = 2)
plot(1/(1 + exp(x_test[, 3]**2 - x_test[, 4]**2 + x_test[, 5])), pred_cont_bag[, 1], pch = 3,
     xlab = "True prob.", ylab = "Pred. prob.", main = "Bagged-DNN")
abline(0, 1, col = 2)
plot(x_test[, 1]**2 + x_test[, 2]**2, pred_cont[, 2], pch = 3,
     xlab = "True lambda", ylab = "Pred. lambda", main = "Single-DNN")
abline(0, 1, col = 2)
plot(x_test[, 1]**2 + x_test[, 2]**2, pred_cont_bag[, 2], pch = 3,
     xlab = "True lambda", ylab = "Pred. lambda", main = "Bagged-DNN")
abline(0, 1, col = 2)

