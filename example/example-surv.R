library(deepTL)
library(MASS)
n <- 5000
p <- 10
n_test <- 10000

par(mfrow = 2:3)
n_ensemble <- 10
esCtrl1 <- list(n.hidden = c(50, 40, 30),
                l1.reg = 10**-4,
                # l2.reg = 10**-2,
                activate = "relu",
                n.batch = 100, n.epoch = 100, early.stop.det = 1000,
                plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam")
esCtrl2 <- esCtrl1
esCtrl2$n.epoch <- 100
esCtrl2$n.batch <- 25

lambda_max <- 5
r <- 0.5
x <- matrix(runif(n*p), n, p)*2-1
h_x <- x[, 1] + 2*x[, 2] #log(lambda_max)*exp(-(x[, 1]**2+x[, 2]**2)/2/r**2) #(x[, 1]**2+x[, 2]**2)/2/r**2 #
# u <- rexp(n, rate = 5)
# t_death <- u/exp(h_x)
t_death <- rexp(n, lambda_max*exp(h_x))
t0 <- rexp(n, rate = lambda_max*mean(exp(h_x)))
y <- apply(cbind(t_death, t0), 1, min)
e <- ifelse(t0 > t_death, 1, 0)

x_test <- matrix(runif(n_test*p), n_test, p)*2-1
h_x_test <- x_test[, 1] + 2*x_test[, 2] #log(lambda_max)*exp(-(x_test[, 1]**2+x_test[, 2]**2)/2/r**2) #(x_test[, 1]**2+x_test[, 2]**2)/2/r**2 #
# u <- rexp(n, rate = 5)
# t_death <- u/exp(h_x)
t_death_test <- rexp(n_test, lambda_max*exp(h_x_test))
t0_test <- rexp(n_test, rate = lambda_max*mean(exp(h_x)))
y_test <- apply(cbind(t_death_test, t0_test), 1, min)
e_test <- ifelse(t0_test > t_death_test, 1, 0)

dnn_dat <- importDnnetSurv(x, y, e)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
args_dnnet <- appendArg(appendArg(esCtrl1, "train", dnn_spl$train, TRUE), "validate", dnn_spl$valid, TRUE)
dnn_mod_surv <- do.call(dnnet, args_dnnet)
# pred_cont <- predict(dnn_mod_cont, x_test)

pred_hx <- predict(dnn_mod_surv, x_test)
plot(pred_hx, h_x_test)
cor(pred_hx, h_x_test)
mean((pred_hx - mean(pred_hx) - (h_x_test - mean(h_x_test)))**2)

plot(x_test[, 1], x_test[, 2], col = gray((h_x_test-min(h_x_test, pred_hx))/(max(h_x_test, pred_hx)-min(h_x_test, pred_hx))), pch = 19, cex = 2)
plot(x_test[, 1], x_test[, 2], col = gray((pred_hx-min(h_x_test, pred_hx))/(max(h_x_test, pred_hx)-min(h_x_test, pred_hx))), pch = 19, cex = 2)
# hist(t_death)
# hist(t0)

bag_mod_surv_b <- ensemble_dnnet(dnn_dat, n_ensemble, esCtrl = esCtrl1)
pred_hx_b <- predict(bag_mod_surv_b, x_test)
plot(pred_hx_b, h_x_test)
cor(pred_hx_b, h_x_test)
mean((pred_hx_b - mean(pred_hx_b) - (h_x_test - mean(h_x_test)))**2)

plot(x_test[, 1], x_test[, 2], col = gray((h_x_test-min(h_x_test))/max(c(h_x_test-min(h_x_test), pred_hx_b-min(pred_hx_b)))), pch = 19, cex = 2)
plot(x_test[, 1], x_test[, 2], col = gray((pred_hx_b-min(pred_hx_b))/max(c(h_x_test-min(h_x_test), pred_hx_b-min(pred_hx_b)))), pch = 19, cex = 2)

# ####
# c_index <- function(y, e, h) {
#
#   y_order <- order(y)
#   y <- y[y_order]
#   e <- e[y_order]
#   h <- h[y_order]
#
#   y <- y[e == 1]
#   h <- h[e == 1]
#   c <- 0
#   count <- 0
#   for(i in 1:(length(y)-1)) {
#
#     c <- c + sum(h[i] < h[(i+1):length(y)])
#     count <- count + length(y) - i
#   }
#     # if(e[i] == 1) {
#     #
#     #   c <- c + sum(h[i] < h[(i+1):length(y)])
#     #   count <- count + length(y) - i
#     # }
#   # print(c(c, count))
#   c/count
# }
#
# c_index(y_test, e_test, pred_hx)
# c_index(y_test, e_test, pred_hx_b)
# c_index(y_test, e_test, h_x_test)
# c_index(y_test, e_test, rnorm(n_test))
#
# c_index(y, e, predict(dnn_mod_surv, x))
# c_index(y, e, predict(bag_mod_surv_b, x))
# c_index(y, e, h_x)

library(survcomp)
c_index_dnn <- concordance.index(pred_hx, y_test, e_test, method = "noether")
c_index_bag <- concordance.index(pred_hx_b, y_test, e_test, method = "noether")
c_index_opt <- concordance.index(h_x_test, y_test, e_test, method = "noether")

library(randomForestSRC)
rf_surv <- rfsrc(Surv(y, e) ~ ., data = data.frame(y, e, x), ntree = 100)


