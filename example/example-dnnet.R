n <- 1000
p <- 5
n_test <- 10000

#### regression
x <- matrix(rnorm(n*p), n, p)
y <- rowMeans(x**2) + rnorm(n)

x_test <- matrix(rnorm(n_test*p), n, p)
y_test <- rowMeans(x_test**2) + rnorm(n_test)

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_cont <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_cont <- predict(dnn_mod_cont, x_test)
mean((pred_cont - y_test)**2)

bag_mod_cont <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_cont_bag <- predict(bag_mod_cont, x_test)
mean((pred_cont_bag - y_test)**2)

mean((y_test - rowMeans(x_test**2))**2)
#### binary classification (with weights)
x <- matrix(rnorm(n*p), n, p)
y1 <- ifelse(rowMeans(x**2) > 1, 1, 0)
y2 <- ifelse(rowMeans(x**2) > 1, 0, 1)
y <- factor(ifelse(c(y1[1:(n/2)], y2[n/2+1:(n/2)]), "A", "B"), levels = c("A", "B"))
w <- c(rep(10, n/2), rep(1, n/2))

x_test <- matrix(rnorm(n_test*p), n, p)
y_test <- factor(ifelse(rowMeans(x_test**2) > 1, "A", "B"), levels = c("A", "B"))

dnn_dat <- importDnnet(x, y, w)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_bnry <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_bnry <- predict(dnn_mod_bnry, x_test)
table(factor(ifelse(pred_bnry[, "A"] > 0.5, "A", "B")), y_test)

bag_mod_bnry <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_gnry_bag <- predict(bag_mod_bnry, x_test)
table(factor(ifelse(pred_gnry_bag[, "A"] > 0.5, "A", "B")), y_test)

#### multi classification
x <- matrix(rnorm(n*p), n, p)
h <- apply(x, 1, max)
y <- factor(ifelse(h < 0.71, "A", ifelse(h < 1.13, "B", ifelse(h < 1.61, "C", "D"))),
            levels = c("A", "B", "C", "D"))

x_test <- matrix(rnorm(n_test*p), n_test, p)
h_test <- apply(x_test, 1, max)
y_test <- factor(ifelse(h_test < 0.71, "A", ifelse(h_test < 1.13, "B", ifelse(h_test < 1.61, "C", "D"))),
                 levels = c("A", "B", "C", "D"))

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_mod_mult <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4,
                      n.batch = 50, n.epoch = 500, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_mult <- predict(dnn_mod_mult, x_test)
table(apply(pred_mult, 1, which.max), y_test)
sum(diag(table(apply(pred_mult, 1, which.max), y_test)))/sum(table(apply(pred_mult, 1, which.max), y_test))

bag_mod_mult <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_multi_bag <- predict(bag_mod_mult, x_test)
table(apply(pred_multi_bag, 1, which.max), y_test)
sum(diag(table(apply(pred_multi_bag, 1, which.max), y_test)))/sum(table(apply(pred_multi_bag, 1, which.max), y_test))

#### ordinal
# x <- matrix(rnorm(n*p), n, p)
# y <- factor(ifelse(apply(x, 1, max) > 1.5,
#                    "A", ifelse(apply(x, 1, max) < 0.5, "C", "B")),
#             levels = c("A", "B", "C"), ordered = TRUE)
#
# x_test <- matrix(rnorm(n_test*p), n, p)
# y_test <- factor(ifelse(apply(x_test, 1, max) > 1.5,
#                         "A", ifelse(apply(x_test, 1, max) < 0.5, "C", "B")),
#                  levels = c("A", "B", "C"), ordered = TRUE)

dnn_dat <- importDnnet(x, factor(y, ordered = TRUE))
dnn_spl <- splitDnnet(dnn_dat, dnn_spl$split)
dnn_mod_ordi <- dnnet(dnn_spl$train, validate = dnn_spl$valid,
                      n.hidden = c(30, 20, 10),
                      l1.reg = 10**-4,
                      n.batch = 50, n.epoch = 500, early.stop.det = 1000, plot = TRUE,
                      accel = "rcpp", learning.rate.adaptive = "adam")
pred_ordi <- predict(dnn_mod_ordi, x_test)
table(apply(pred_ordi, 1, which.max), y_test)
sum(diag(table(apply(pred_ordi, 1, which.max), y_test)))/sum(table(apply(pred_ordi, 1, which.max), y_test))

bag_mod_ordi <- ensemble_dnnet(dnn_dat, 10,
                               esCtrl = list(n.hidden = c(30, 20, 10),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                                             plot = TRUE,
                                             accel = "rcpp", learning.rate.adaptive = "adam"))
pred_ordi_bag <- predict(bag_mod_ordi, x_test)
table(apply(pred_ordi_bag, 1, which.max), y_test)
sum(diag(table(apply(pred_ordi_bag, 1, which.max), y_test)))/sum(table(apply(pred_ordi_bag, 1, which.max), y_test))




