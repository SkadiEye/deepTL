library(MASS)
# library(deepTL)
library(randomForest)
library(glmnet)
library(devtools)
load_all()

cor_unif <- function(n, mu, sigma, a = 0, b = 1) {

  x_norm <- mvrnorm(n, mu, sigma)
  apply((x_norm - rep(1, n) %*% t(mu))/(rep(1, n) %*% t(sqrt(diag(sigma)))), 2, pnorm)*(b - a) + a
}

n <- 500
p <- 100
n_path <- 10
rho <- 0.4
n_test <- 10000

x <- matrix(0, n, p*n_path)
for(i in 1:n_path)
  x[, (i-1)*p+1:p] <- cor_unif(n, mu = rep(0, p), sigma = diag(p)*(1-rho)+rho, a = -1, b = 1)
y <- x[, 1]**2*5 + (x[, p+1] - x[, p+2])*2 + (x[, p*2+1] + x[, p*2+2] + x[, p*2+3])*x[, p*3+1]*3 + rnorm(n)

x_test <- matrix(0, n_test, p*n_path)
for(i in 1:n_path)
  x_test[, (i-1)*p+1:p] <- cor_unif(n_test, mu = rep(0, p), sigma = diag(p)*(1-rho)+rho, a = -1, b = 1)
y_test <- x_test[, 1]**2*5 + (x_test[, p+1] - x_test[, p+2])*2 +
  (x_test[, p*2+1] + x_test[, p*2+2] + x_test[, p*2+3])*x_test[, p*3+1]*3 + rnorm(n_test)

par(mfrow = 2:3)
n_ensemble <- 10
esCtrl <- list(n.hidden = c(50, 40, 30, 20),
               l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
               plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam")

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)

# dnn_mod <- ensemble_dnnet(dnn_dat, n_ensemble, esCtrl = esCtrl)
# dnn_pred <- predict(dnn_mod, x_test)

pathway.list <- list()
for(i in 1:n_path) pathway.list[[i]] <- (i-1)*p+1:p
dnn_mod0 <- dnnet(dnn_spl$train, validate = dnn_spl$valid, n.hidden = c(50, 40, 30, 20),
                  l1.reg = 10**-4, n.batch = 50, n.epoch = 1000, early.stop.det = 1000,
                  plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam",
                  pathway = TRUE, pathway.list = pathway.list, l1.pathway = 0.001,
                  pathway.active = "tanh")
dnn0_pred <- predict(dnn_mod0, x_test)
cor(y_test, dnn0_pred)
mean((y_test - dnn0_pred)**2)
plot(y_test, dnn0_pred)
abline(0, 1, col = 2)

weights0 <- as.data.frame(dnn_mod0@model.spec$weight.pathway)
names(weights0) <- 1:10
round(weights0, 3)

dnn_mod1 <- dnnet(dnn_spl$train, validate = dnn_spl$valid, n.hidden = c(50, 40, 30, 20),
                  l1.reg = 10**-4, n.batch = 50, n.epoch = 1500, early.stop.det = 1000,
                  plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam")
dnn1_pred <- predict(dnn_mod1, x_test)
cor(y_test, dnn1_pred)
mean((y_test - dnn1_pred)**2)
plot(y_test, dnn1_pred)
abline(0, 1, col = 2)

###################
n <- 1000
p <- 4
n_path <- 100
n_test <- 10000

x <- matrix(0, n, p*n_path)
for(j in 1:n_path)
  for(i in 1:n)
    x[i, sample(1:4, 1)+(j-1)*4] <- 1
y <- x[, 3]*3 + x[, 5]*x[, 10]*6 + (x[, 15]*3 - 1)*(x[, 20]*2 + 1) + exp(x[, 2] + x[, 7]) + rnorm(n)

x_test <- matrix(0, n_test, p*n_path)
for(j in 1:n_path)
  for(i in 1:n_test)
    x_test[i, sample(1:4, 1)+(j-1)*4] <- 1
y_test <- x_test[, 3]*3 + x_test[, 5]*x_test[, 10]*6 + (x_test[, 15]*3 - 1)*(x_test[, 20]*2 + 1) +
  exp(x_test[, 2] + x_test[, 7]) + rnorm(n_test)

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)

pathway.list <- list()
for(i in 1:n_path) pathway.list[[i]] <- (i-1)*p+1:p
dnn_mod0 <- dnnet(dnn_spl$train, validate = dnn_spl$valid, n.hidden = c(50, 40, 30, 20),
                  l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                  plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam",
                  pathway = TRUE, pathway.list = pathway.list, l2.pathway = 0.01, # l1.pathway = 0.001,
                  pathway.active = "identity")
dnn0_pred <- predict(dnn_mod0, x_test)
cor(y_test, dnn0_pred)
mean((y_test - dnn0_pred)**2)
plot(y_test, dnn0_pred)
abline(0, 1, col = 2)

weights0 <- as.data.frame(dnn_mod0@model.spec$weight.pathway)
names(weights0) <- 1:10
round(weights0, 3)

dnn_mod1 <- dnnet(dnn_spl$train, validate = dnn_spl$valid, n.hidden = c(50, 40, 30, 20),
                  l1.reg = 10**-4, n.batch = 50, n.epoch = 500, early.stop.det = 1000,
                  plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam")
dnn1_pred <- predict(dnn_mod1, x_test)
cor(y_test, dnn1_pred)
mean((y_test - dnn1_pred)**2)
plot(y_test, dnn1_pred)
abline(0, 1, col = 2)

####
path_ <- "C:/Users/xm2231/Dropbox (UFL)/permuteBaggedDNN/real-data-ctou/Cummings and Myers (2004)/"

dat1 <- read.table(paste0(path_, "12859_2004_248_MOESM1_ESM.txt"), header = TRUE)
dat2 <- read.table(paste0(path_, "12859_2004_248_MOESM2_ESM.txt"), header = TRUE)
dat3 <- read.table(paste0(path_, "12859_2004_248_MOESM3_ESM.txt"), header = TRUE)
dat4 <- read.table(paste0(path_, "12859_2004_248_MOESM4_ESM.txt"), header = TRUE)

dat_all <- rbind(data.frame(from = "a", dat1),
                 data.frame(from = "b", dat2),
                 data.frame(from = "c", dat3))

mod_glm <- glm(edit ~ ., dat_all[, -c(1, 23, 47)], family = "binomial")
pred_glm <- predict(mod_glm, dat_all[, -c(1, 23, 47)], type = "response")
mean(dat_all$edit != ifelse(pred_glm > 0.5, levels(dat_all$edit)[1], levels(dat_all$edit)[2]))

mod_glm <- glm(edit ~ ., dat_all[, -c(23)], family = "binomial")
pred_glm <- predict(mod_glm, dat_all[, -c(23)], type = "response")
mean(dat_all$edit != ifelse(pred_glm > 0.5, levels(dat_all$edit)[1], levels(dat_all$edit)[2]))

mod_glm <- glm(from ~ ., dat_all[, -c(2, 23)], family = "binomial")
pred_glm <- predict(mod_glm, dat_all[, -c(2, 23)], type = "response")
mean(dat_all$edit != ifelse(pred_glm > 0.5, levels(dat_all$edit)[1], levels(dat_all$edit)[2]))

mat_ <- model.matrix(mod_glm)

x <- mat_[, -1]
y <- dat_all$edit
pathway.list <- list()
for(i in 1:20)
  pathway.list[[i]] <- 1:4+(i-1)*4
pathway.list[[21]] <- 81:83
pathway.list[[22]] <- 84:86
for(i in 23:40)
  pathway.list[[i]] <- 1:4+(i-1)*4-2
pathway.list[[41]] <- 159:161

dnn_dat <- importDnnet(x, y)
dnn_spl <- splitDnnet(dnn_dat, 0.8)
dnn_spl_train <- splitDnnet(dnn_spl$train, 0.8)

dnn_mod0 <- dnnet(dnn_spl_train$train, validate = dnn_spl_train$valid, n.hidden = c(50, 40, 30, 20), # c(100, 75, 50, 25), #
                  l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                  plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam",
                  pathway = TRUE, pathway.list = pathway.list, l1.pathway = 0.0001,
                  pathway.active = "identity")
dnn0_pred <- predict(dnn_mod0, dnn_spl$valid@x)
table(colnames(dnn0_pred)[apply(dnn0_pred, 1, which.max)], dnn_spl$valid@y)
mean(colnames(dnn0_pred)[apply(dnn0_pred, 1, which.max)] == dnn_spl$valid@y)

dnn_mod0_bag <- ensemble_dnnet(dnn_spl$train, 10, esCtrl = list(n.hidden = c(50, 40, 30, 20),
                               l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                               plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam",
                               pathway = TRUE, pathway.list = pathway.list, l1.pathway = 0.0001,
                               pathway.active = "tanh"))
dnn0_pred_bag <- predict(dnn_mod0_bag, dnn_spl$valid@x)
table(colnames(dnn0_pred_bag)[apply(dnn0_pred_bag, 1, which.max)], dnn_spl$valid@y)
mean(colnames(dnn0_pred_bag)[apply(dnn0_pred_bag, 1, which.max)] == dnn_spl$valid@y)

dnn_mod1 <- dnnet(dnn_spl$train, validate = dnn_spl$valid, n.hidden = c(50, 40, 30, 20),
                  l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                  plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam")
dnn1_pred <- predict(dnn_mod1, dnn_spl$valid@x)
table(colnames(dnn1_pred)[apply(dnn1_pred, 1, which.max)], dnn_spl$valid@y)
mean(colnames(dnn1_pred)[apply(dnn1_pred, 1, which.max)] == dnn_spl$valid@y)

dnn_mod1_bag <- ensemble_dnnet(dnn_spl$train, 10,
                               esCtrl = list(n.hidden = c(50, 40, 30, 20),
                                             l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                                             plot = TRUE, accel = "rcpp", learning.rate.adaptive = "adam"))
dnn1_pred_bag <- predict(dnn_mod1_bag, dnn_spl$valid@x)
table(colnames(dnn1_pred_bag)[apply(dnn1_pred_bag, 1, which.max)], dnn_spl$valid@y)
mean(colnames(dnn1_pred_bag)[apply(dnn1_pred_bag, 1, which.max)] == dnn_spl$valid@y)

rf_mod <- randomForest(dat_all[dnn_spl$split, -c(1, 2, 23)], dat_all$edit[dnn_spl$split], ntree = 1000)
rf_pred <- predict(rf_mod, dat_all[-dnn_spl$split, -c(1, 2, 23)])
table(rf_pred, dnn_spl$valid@y)
mean(rf_pred == dnn_spl$valid@y)

#### RF
rf_mod <- randomForest(x, y, ntree = 1000)
rf_pred <- predict(rf_mod, x_test)

#### LASSO
cv_lasso <- cv.glmnet(x, y, family = "gaussian")
lasso_mod <- glmnet(x, y, family = "gaussian", lambda = cv_lasso$lambda[which.min(cv_lasso$cvm)])
lasso_pred <- predict(lasso_mod, x_test)[, "s0"]

# cor(y_test, dnn_pred)
cor(y_test, rf_pred)
cor(y_test, lasso_pred)

# mean((y_test - dnn_pred)**2)
mean((y_test - rf_pred)**2)
mean((y_test - lasso_pred)**2)

# 1 - mean((y_test - dnn_pred)**2)/mean((y_test - mean(y_test))**2)
1 - mean((y_test - rf_pred)**2)/mean((y_test - mean(y_test))**2)
1 - mean((y_test - lasso_pred)**2)/mean((y_test - mean(y_test))**2)




