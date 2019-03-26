library(deepTL)
library(MASS)
n <- 1000
p <- 5
n_test <- 10000

par(mfrow = 2:3)
#### Example I: Regression
x <- matrix(rnorm(n*p), n, p)
y <- rowMeans(x**2) + rnorm(n)

x_test <- matrix(rnorm(n_test*p), n_test, p)
y_test <- rowMeans(x_test**2) + rnorm(n_test)

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_cont <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_cont <- predict(dnn_mod_cont, x_test)
# mean((pred_cont - y_test)**2)

bag_mod_cont <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_cont_bag <- predict(bag_mod_cont, x_test)
# mean((pred_cont_bag - y_test)**2)

cat("------- Example I: Regression ------- \n",
    "      True MSE:", round(mean((y_test - rowMeans(x_test**2))**2), 4), "\n",
    "       DNN MSE:", round(mean((pred_cont - y_test)**2), 4), "\n",
    "Bagged DNN MSE:", round(mean((pred_cont_bag - y_test)**2), 4), "\n")

#### Binary classification I (with weights)
x <- matrix(rnorm(n*p), n, p)
y1 <- ifelse(rowMeans(x**2) > 1, 1, 0)
y2 <- ifelse(rowMeans(x**2) > 1, 0, 1)
y <- factor(ifelse(c(y1[1:(n/2)], y2[n/2+1:(n/2)]), "A", "B"), levels = c("A", "B"))
w <- c(rep(10, n/2), rep(1, n/2))

x_test <- matrix(rnorm(n_test*p), n_test, p)
y_test <- factor(ifelse(rowMeans(x_test**2) > 1, "A", "B"), levels = c("A", "B"))

dnn_dat <- importDnnet(x, y, w)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_bnry <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_bnry <- predict(dnn_mod_bnry, x_test)
# table(factor(ifelse(pred_bnry[, "A"] > 0.5, "A", "B")), y_test)

bag_mod_bnry <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_gnry_bag <- predict(bag_mod_bnry, x_test)
# table(factor(ifelse(pred_gnry_bag[, "A"] > 0.5, "A", "B")), y_test)

cat("------- Example II: Binary Classification (with weights) ------- \n",
    "      True mis-class rate:", 0, "\n",
    "       DNN mis-class rate:", round(mean(factor(ifelse(pred_bnry[, "A"] > 0.5, "A", "B")) != y_test), 4), "\n",
    "Bagged DNN mis-class rate:", round(mean(factor(ifelse(pred_gnry_bag[, "A"] > 0.5, "A", "B")) != y_test), 4), "\n")

#### Binary classification II
x <- matrix(runif(n*p), n, p)*2-1
prob <- 1/(1+exp(15 - (rowSums(x**2))**4))
y <- factor(ifelse(runif(n) < prob, "A", "B"), levels = c("A", "B"))

x_test <- matrix(runif(n_test*p), n_test, p)
prob_test <- 1/(1+exp(15 - (rowSums(x_test**2))**4))
y_test <- factor(ifelse(runif(n_test) < prob_test, "A", "B"), levels = c("A", "B"))

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_bnry <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_bnry <- predict(dnn_mod_bnry, x_test)
# table(factor(ifelse(pred_bnry[, "A"] > 0.5, "A", "B")), y_test)

bag_mod_bnry <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_gnry_bag <- predict(bag_mod_bnry, x_test)
# table(factor(ifelse(pred_gnry_bag[, "A"] > 0.5, "A", "B")), y_test)

cat("------- Example III: Binary Classification ------- \n",
    "      True mis-class rate:", round(mean(factor(ifelse(prob_test > 0.5, "A", "B")) != y_test), 4), "\n",
    "       DNN mis-class rate:", round(mean(factor(ifelse(pred_bnry[, "A"] > 0.5, "A", "B")) != y_test), 4), "\n",
    "Bagged DNN mis-class rate:", round(mean(factor(ifelse(pred_gnry_bag[, "A"] > 0.5, "A", "B")) != y_test), 4), "\n")

#### Multi classification I
x <- matrix(runif(n*p), n, p)*2-1
prob <- exp(x**2*10)/rowSums(exp(x**2*10))
y <- factor(apply(prob, 1, function(x) which.max(rmultinom(1, 1, x))))

x_test <- matrix(runif(n_test*p), n_test, p)*2-1
prob_test <- exp(x_test**2*10)/rowSums(exp(x_test**2*10))
y_test <- factor(apply(prob_test, 1, function(x) which.max(rmultinom(1, 1, x))))

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_mult <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4,
                      n.batch = 50, n.epoch = 500, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_mult <- predict(dnn_mod_mult, x_test)
# table(apply(pred_mult, 1, which.max), y_test)
# sum(diag(table(apply(pred_mult, 1, which.max), y_test)))/sum(table(apply(pred_mult, 1, which.max), y_test))

bag_mod_mult <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_multi_bag <- predict(bag_mod_mult, x_test)
# table(apply(pred_multi_bag, 1, which.max), y_test)
# sum(diag(table(apply(pred_multi_bag, 1, which.max), y_test)))/sum(table(apply(pred_multi_bag, 1, which.max), y_test))

#### ordinal
dnn_dat <- importDnnet(x, factor(y, ordered = TRUE))
dnn_spl <- splitDnnet(dnn_dat, dnn_spl$split)
dnn_mod_ordi <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4,
                      n.batch = 50, n.epoch = 500, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_ordi <- predict(dnn_mod_ordi, x_test)
# table(apply(pred_ordi, 1, which.max), y_test)
# sum(diag(table(apply(pred_ordi, 1, which.max), y_test)))/sum(table(apply(pred_ordi, 1, which.max), y_test))

bag_mod_ordi <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_ordi_bag <- predict(bag_mod_ordi, x_test)
# table(apply(pred_ordi_bag, 1, which.max), y_test)
# sum(diag(table(apply(pred_ordi_bag, 1, which.max), y_test)))/sum(table(apply(pred_ordi_bag, 1, which.max), y_test))

cat("------- Example IV: Multi Classification I ------- \n",
    "                True mis-class rate:", 0, "\n",
    "         DNN (Multi) mis-class rate:", round(1-sum(diag(table(apply(pred_mult, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_mult, 1, which.max), y_test)), 4), "\n",
    "  Bagged DNN (Multi) mis-class rate:", round(1-sum(diag(table(apply(pred_multi_bag, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_multi_bag, 1, which.max), y_test)), 4), "\n",
    "       DNN (Ordinal) mis-class rate:", round(1-sum(diag(table(apply(pred_ordi, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_ordi, 1, which.max), y_test)), 4), "\n",
    "Bagged DNN (Ordinal) mis-class rate:", round(1-sum(diag(table(apply(pred_ordi_bag, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_ordi_bag, 1, which.max), y_test)), 4), "\n")

#### Multi classification II (Ordinal)
x <- matrix(runif(n*p), n, p)*2-1
prob <- -9*(matrix(rowSums(x**3), n, 5) - rep(1, n) %*% t(c(-1, -0.5, 0, 0.5, 1)))**2
prob <- exp(prob)/rowSums(exp(prob))
y <- factor(apply(prob, 1, function(x) which.max(rmultinom(1, 1, x))))

x_test <- matrix(runif(n_test*p), n_test, p)*2-1
prob_test <- -9*(matrix(rowSums(x_test**3), n_test, 5) - rep(1, n_test) %*% t(c(-1, -0.5, 0, 0.5, 1)))**2
prob_test <- exp(prob_test)/rowSums(exp(prob_test))
y_test <- factor(apply(prob_test, 1, function(x) which.max(rmultinom(1, 1, x))))

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_mult <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4,
                      n.batch = 50, n.epoch = 500, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_mult <- predict(dnn_mod_mult, x_test)
# table(apply(pred_mult, 1, which.max), y_test)
# sum(diag(table(apply(pred_mult, 1, which.max), y_test)))/sum(table(apply(pred_mult, 1, which.max), y_test))

bag_mod_mult <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_multi_bag <- predict(bag_mod_mult, x_test)
# table(apply(pred_multi_bag, 1, which.max), y_test)
# sum(diag(table(apply(pred_multi_bag, 1, which.max), y_test)))/sum(table(apply(pred_multi_bag, 1, which.max), y_test))

#### ordinal
dnn_dat <- importDnnet(x, factor(y, ordered = TRUE))
dnn_spl <- splitDnnet(dnn_dat, dnn_spl$split)
dnn_mod_ordi <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4,
                      n.batch = 50, n.epoch = 500, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_ordi <- predict(dnn_mod_ordi, x_test)
# table(apply(pred_ordi, 1, which.max), y_test)
# sum(diag(table(apply(pred_ordi, 1, which.max), y_test)))/sum(table(apply(pred_ordi, 1, which.max), y_test))

bag_mod_ordi <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_ordi_bag <- predict(bag_mod_ordi, x_test)
# table(apply(pred_ordi_bag, 1, which.max), y_test)
# sum(diag(table(apply(pred_ordi_bag, 1, which.max), y_test)))/sum(table(apply(pred_ordi_bag, 1, which.max), y_test))

cat("------- Example V: Multi Classification II (Ordinal) ------- \n",
    "                True mis-class rate:", round(mean(apply(prob_test, 1, which.max) != y_test), 4), "|",
    "error:", round(mean(abs(apply(prob_test, 1, which.max) - as.numeric(y_test))), 4), "\n",
    "         DNN (Multi) mis-class rate:", round(1-sum(diag(table(apply(pred_mult, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_mult, 1, which.max), y_test)), 4), "|",
    "error:", round(mean(abs(apply(pred_mult, 1, which.max) - as.numeric(y_test))), 4), "\n",
    "  Bagged DNN (Multi) mis-class rate:", round(1-sum(diag(table(apply(pred_multi_bag, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_multi_bag, 1, which.max), y_test)), 4), "|",
    "error:", round(mean(abs(apply(pred_multi_bag, 1, which.max) - as.numeric(y_test))), 4), "\n",
    "       DNN (Ordinal) mis-class rate:", round(1-sum(diag(table(apply(pred_ordi, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_ordi, 1, which.max), y_test)), 4), "|",
    "error:", round(mean(abs(apply(pred_ordi, 1, which.max) - as.numeric(y_test))), 4), "\n",
    "Bagged DNN (Ordinal) mis-class rate:", round(1-sum(diag(table(apply(pred_ordi_bag, 1, which.max), y_test)))/
                                                    sum(table(apply(pred_ordi_bag, 1, which.max), y_test)), 4), "|",
    "error:", round(mean(abs(apply(pred_ordi_bag, 1, which.max) - as.numeric(y_test))), 4), "\n")



