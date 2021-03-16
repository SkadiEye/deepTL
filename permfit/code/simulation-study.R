###################################
#### PermFIT: Simulation Study ####
#### (Continuous)              ####
#### 04/04/2020, Xinlei Mi     ####
###################################

#### Required packages, ignore if installed
require(devtools)
devtools::install_github("SkadiEye/deepTL")
require(randomForest)
require(glmnet)
require(e1071)
require(caret)

#### Load required libraries
library(deepTL)
library(randomForest)
library(glmnet)
library(MASS)
library(e1071)
library(caret)

#### Generate Simulated Data
generate_dat <- function(x, error, p_pathway) {
  x[, 1] + log(1 + (x[, p_pathway*2 + 1] + 1)**2 + x[, p_pathway + 1]**2*2)*2 +
    x[, p_pathway*3 + 1]*x[, p_pathway*4 + 1] + rnorm(dim(x)[1])*error
}
n <- 1000
p <- 100
n_test <- 10000
n_pathway <- 10
rho <- 0.5
sigma_ <- diag(p/n_pathway)*(1-rho) + matrix(rho, p/n_pathway, p/n_pathway)

set.seed(10086)
x <- matrix(NA, n, p)
for(i in 1:n_pathway)
  x[, (i-1)*p/n_pathway + 1:(p/n_pathway)] <- mvrnorm(n, rep(0, p/n_pathway), sigma_)
y <- generate_dat(x, 1, p/n_pathway)

x_test <- matrix(NA, n_test, p)
for(i in 1:n_pathway)
  x_test[, (i-1)*p/n_pathway + 1:(p/n_pathway)] <- mvrnorm(n_test, rep(0, p/n_pathway), sigma_)
y_test <- generate_dat(x_test, 1, p/n_pathway)

#### 1. DNN ####
## 1.0 Hyper-parameters
esCtrl <- list(n.hidden = c(50, 40, 30, 20), activate = "relu",
               l1.reg = 10**-4, early.stop.det = 1000, n.batch = 50,
               n.epoch = 1000, learning.rate.adaptive = "adam", plot = FALSE)
esCtrl$n.epoch <- 200
n_ensemble <- 100
n_perm <- 10
## Attention: the parameters above are for faster implementation
## Attention: the parameters below are used in the paper
# Set esCtrl$n.epoch <- 1000
# n_ensemble <- 100
# n_perm <- 100
## 1.1 DNN (full model) and performance
dnn_obj <- importDnnet(x = x, y = y)
dnn_mod <- ensemble_dnnet(dnn_obj, n_ensemble, esCtrl)
dnn_pred <- predict(dnn_mod, x_test)
dnn_perf <- data.frame(method = "DNN",
                       mse = mean((dnn_pred - y_test)**2),
                       cor = cor(dnn_pred, y_test))
## 1.2 PermFIT-DNN
shuffle <- sample(n)
dat_spl <- splitDnnet(dnn_obj, 0.8)
permfit_dnn <- permfit(train = dat_spl$train, validate = dat_spl$valid, k_fold = 0,
                       pathway_list = list(), n_perm = n_perm,
                       method = "ensemble_dnnet", shuffle = shuffle,
                       n.ensemble = n_ensemble, esCtrl = esCtrl)
## 1.2.1 Re-fit with PermFIT-DNN features
imp_feature <- which(permfit_dnn@importance$importance_pval < 0.1)
dnn_obj0 <- importDnnet(x = x[, imp_feature], y = y)
dnn_mod0 <- ensemble_dnnet(dnn_obj0, n_ensemble, esCtrl)
dnn_pred0 <- predict(dnn_mod0, x_test[, imp_feature])
dnn_perf0 <- data.frame(method = "PermFIT-DNN",
                        mse = mean((dnn_pred0 - y_test)**2),
                        cor = cor(dnn_pred0, y_test))

#### 2. SVM ####
## 2.1 SVM (full model) and performance
svm_mod <- tune.svm(x, y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                    tunecontrol = tune.control(cross = 5))
svm_pred <- predict(svm_mod$best.model, x_test)
svm_perf <- data.frame(method = "SVM",
                       mse = mean((svm_pred - y_test)**2),
                       cor = cor(svm_pred, y_test))
## 2.2 PermFIT-SVM
permfit_svm <- permfit(train = dat_spl$train, validate = dat_spl$valid, k_fold = 0,
                       pathway_list = list(), n_perm = n_perm,
                       method = "svm", shuffle = shuffle)
## 2.2.1 Re-fit with PermFIT-SVM features
imp_feature <- which(permfit_svm@importance$importance_pval < 0.1)
svm_mod0 <- tune.svm(x[, imp_feature], y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                     tunecontrol = tune.control(cross = 5))
svm_pred0 <- predict(svm_mod0$best.model, x_test[, imp_feature])
svm_perf0 <- data.frame(method = "PermFIT-SVM",
                        mse = mean((svm_pred0 - y_test)**2),
                        cor = cor(svm_pred0, y_test))
## 2.3 SVM-RFE (Attention: RFE is slow)
colnames(x) <- paste0("x", 1:dim(x)[2])
svmProfile <- rfe(x, y, sizes = c(5, 10, 15),
                  rfeControl = rfeControl(functions = caretFuncs, number = 100),
                  method = "svmRadial")
## 2.3.1 Re-fit with top 10 SVM-RFE features
scorex <- aggregate(svmProfile$variables$Overall, list(var = svmProfile$variables$var), mean)
svm_dVal <- scorex$x[match(colnames(x), scorex$var)]
svm_rank <- rank(-svm_dVal)
imp_feature <- which(svm_rank <= 10)
svm_mod1 <- tune.svm(x[, imp_feature], y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                     tunecontrol = tune.control(cross = 5))
svm_pred1 <- predict(svm_mod1$best.model, x_test[, imp_feature])
svm_perf1 <- data.frame(method = "RFE-SVM",
                        mse = mean((svm_pred1 - y_test)**2),
                        cor = cor(svm_pred1, y_test))

#### 3. RF ####
## 3.1 RF (full model) and performance
rf_mod <- randomForest(x, y, ntree = 1000, importance = TRUE)
rf_pred <- predict(rf_mod, x_test)
rf_perf <- data.frame(method = "RF",
                      mse = mean((rf_pred - y_test)**2),
                      cor = cor(rf_pred, y_test))
## 3.2 PermFIT-RF
permfit_rf <- permfit(train = dat_spl$train, validate = dat_spl$valid, k_fold = 0,
                      pathway_list = list(), n_perm = n_perm,
                      method = "random_forest", shuffle = shuffle, ntree = 1000)
## 3.2.1 Re-fit with PermFIT-RF features
imp_feature <- which(permfit_rf@importance$importance_pval < 0.1)
rf_mod0 <- randomForest(x[, imp_feature], y, ntree = 1000, importance = TRUE)
rf_pred0 <- predict(rf_mod0, x_test[, imp_feature])
rf_perf0 <- data.frame(method = "PermFIT-RF",
                       mse = mean((rf_pred0 - y_test)**2),
                       cor = cor(rf_pred0, y_test))
## 3.3 Vanilla-RF
rf_imp <- rf_mod$importance[, "%IncMSE"]
rf_pval <- 1 - pnorm(rf_imp/rf_mod$importanceSD)
## 3.3.1 Re-fit with Vanilla-RF features
imp_feature <- which(rf_pval < 0.1)
rf_mod1 <- randomForest(x[, imp_feature], y, ntree = 1000, importance = TRUE)
rf_pred1 <- predict(rf_mod1, x_test[, imp_feature])
rf_perf1 <- data.frame(method = "Vanilla-RF",
                       mse = mean((rf_pred1 - y_test)**2),
                       cor = cor(rf_pred1, y_test))
#### 4. Summary
## 4.1 Importance scores and p-values
data.frame(varname = paste0("X", 1:p, ifelse((1:p) %% (p/n_pathway) == 1 & (1:p) < p/2, "***", "")),
           `PermFIT-DNN` = paste0(round(permfit_dnn@importance$importance, 3), " (p-value = ",
                                  round(permfit_dnn@importance$importance_pval, 3), ")"),
           `PermFIT-SVM` = paste0(round(permfit_svm@importance$importance, 3), " (p-value = ",
                                  round(permfit_svm@importance$importance_pval, 3), ")"),
           `PermFIT-RF` = paste0(round(permfit_rf@importance$importance, 3), " (p-value = ",
                                 round(permfit_rf@importance$importance_pval, 3), ")"),
           check.names = FALSE)
## 4.2 Performace
rbind(dnn_perf, dnn_perf0, rf_perf, rf_perf0, rf_perf1, svm_perf, svm_perf0, svm_perf1)
## Each scenario is randomly repeated 100 times for the simulation study

