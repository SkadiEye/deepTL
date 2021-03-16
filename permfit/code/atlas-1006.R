###################################
#### PermFIT: Microbiome Atlas ####
#### 04/04/2020, Xinlei Mi     ####
###################################

#### Required packages, ignore if installed
require(devtools)
devtools::install_github("SkadiEye/deepTL")
require(randomForest)
require(glmnet)
require(e1071)
require(caret)
require(ROCR)

#### Load required libraries
library(deepTL)
library(randomForest)
library(glmnet)
library(MASS)
library(e1071)
library(caret)
library(ROCR)

#### Load cleaned-data
dat <- readRDS("./cleaned-dat.RDS")
y <- dat$atlas$y
y <- as.numeric(y)
## x is a matrix including only numeric variable (and dummy variables)
## x_rf is a data.frame contraining categorical variables
x <- as.matrix(dat$atlas$x)
x_rf <- dat$atlas$x_rf

#### Functions for Evaluation
mse <- function(y, x) mean((y - x)**2)

#### 0.1 Hyper-parameters ####
esCtrl <- list(n.hidden = c(50, 40, 30, 20), activate = "relu",
               l1.reg = 10**-4, early.stop.det = 1000, n.batch = 50,
               n.epoch = 1000, learning.rate.adaptive = "adam", plot = FALSE)
esCtrl$n.epoch <- 100
n_ensemble <- 10
n_perm <- 10
## Attention: the parameters above are for faster implementation
## Attention: the parameters below are used in the paper
# Set esCtrl$n.epoch <- 1000
# n_ensemble <- 100
# n_perm <- 100
#### 0.2 Random Shuffle
set.seed(20200404)
shuffle <- sample(length(y))

#### 1. Importance ####
#### 1.1 PermFIT-DNN ####
dnn_obj <- importDnnet(x = x, y = y)
permfit_dnn <- permfit(train = dnn_obj, k_fold = 5, n_perm = n_perm,
                       method = "ensemble_dnnet", shuffle = shuffle,
                       n.ensemble = n_ensemble, esCtrl = esCtrl, verbose = 0)
dnn_feature <- which(permfit_dnn@importance$importance_pval < 0.1)
#### 1.2 PermFIT-SVM ####
permfit_svm <- permfit(train = dnn_obj, k_fold = 5, n_perm = n_perm,
                       method = "svm", shuffle = shuffle,
                       n.ensemble = n_ensemble)
svm_feature <- which(permfit_svm@importance$importance_pval < 0.1)
#### 1.3 PermFIT-RF ####
permfit_rf <- permfit(train = dnn_obj, k_fold = 5, n_perm = n_perm,
                      method = "random_forest", shuffle = shuffle,
                      n.ensemble = n_ensemble, ntree = 1000)
rf_feature <- which(permfit_rf@importance$importance_pval < 0.1)
#### 1.4 SVM-RFE #### (Attention: RFE is slow)
svmProfile <- rfe(x, y, sizes = c(5, 10, 15),
                  rfeControl = rfeControl(functions = caretFuncs, number = 100),
                  method = "svmRadial")
scorex <- aggregate(svmProfile$variables$Overall, list(var = svmProfile$variables$var), mean)
svm_dVal <- scorex$x[match(colnames(x), scorex$var)]
svm_rank <- rank(-svm_dVal)
svm_rfe_feature <- which(svm_rank <= 20)
#### 1.5 Vanilla-RF ####
rf_mod <- randomForest(x_rf, y, ntree = 1000, importance = TRUE)
rf_vanilla_feature <- which(1 - pnorm(rf_mod$importance[, "%IncMSE"]/rf_mod$importanceSD) < 0.1)

#### 2. 5-fold Cross-Validation ####
k_fold <- 5
pred <- matrix(NA, length(y), 8)
for(k in 1:k_fold) {

  validate <- permfit_dnn@validation_index[[k]]

  #### 2.1 DNN ####
  dnn_mod <- ensemble_dnnet(importDnnet(x = x[-validate, ], y = y[-validate]), n_ensemble, esCtrl, verbose = 0)
  pred[validate, 1] <- predict(dnn_mod, x[validate, ])

  #### 2.2 PermFIT-DNN ####
  dnn_mod <- ensemble_dnnet(importDnnet(x = x[-validate, dnn_feature], y = y[-validate]), n_ensemble, esCtrl, verbose = 0)
  pred[validate, 2] <- predict(dnn_mod, x[validate, dnn_feature])

  #### 2.3 SVM ####
  svm_mod <- tune.svm(x[-validate, ], y[-validate], gamma = 10**(-(0:4)), cost = 10**(0:4/2), tunecontrol = tune.control(cross = 5))
  pred[validate, 3] <- predict(svm_mod$best.model, x[validate, ])

  #### 2.4 PermFIT-SVM ####
  svm_mod <- tune.svm(x[-validate, svm_feature], y[-validate], gamma = 10**(-(0:4)), cost = 10**(0:4/2), tunecontrol = tune.control(cross = 5))
  pred[validate, 4] <- predict(svm_mod$best.model, x[validate, svm_feature])

  #### 2.5 RFE-SVM ####
  svm_mod <- tune.svm(x[-validate, svm_rfe_feature], y[-validate], gamma = 10**(-(0:4)), cost = 10**(0:4/2), tunecontrol = tune.control(cross = 5))
  pred[validate, 5] <- predict(svm_mod$best.model, x[validate, svm_rfe_feature])

  #### 2.6 RF ####
  rf_mod <- randomForest(x_rf[-validate, ], y[-validate], ntree = 1000, importance = TRUE)
  pred[validate, 6] <- predict(rf_mod, x_rf[validate, ])

  #### 2.7 PermFIT-RF ####
  rf_mod <- randomForest(x_rf[-validate, rf_feature], y[-validate], ntree = 1000, importance = TRUE)
  pred[validate, 7] <- predict(rf_mod, x_rf[validate, rf_feature])

  #### 2.8 Vanilla-RF ####
  rf_mod <- randomForest(x_rf[-validate, rf_vanilla_feature], y[-validate], ntree = 1000, importance = TRUE)
  pred[validate, 8] <- predict(rf_mod, x_rf[validate, rf_vanilla_feature])
}

#### 3. Summary
## 3.1 Importance scores and p-values
data.frame(var_name = colnames(x),
           `PermFIT-DNN` = paste0(round(permfit_dnn@importance$importance, 5), " (p-value = ",
                                  round(permfit_dnn@importance$importance_pval, 3), ")"),
           `PermFIT-SVM` = paste0(round(permfit_svm@importance$importance, 5), " (p-value = ",
                                  round(permfit_svm@importance$importance_pval, 3), ")"),
           `PermFIT-RF` = paste0(round(permfit_rf@importance$importance, 5), " (p-value = ",
                                 round(permfit_rf@importance$importance_pval, 3), ")"),
           check.names = FALSE)
## Note in the paper, PermFIT is repeated 100 times.
## The median of p-values for each variable is selected, followed by FDR control
## 3.2 Performace
data.frame(method = c("DNN", "PermFIT-DNN", "SVM", "PermFIT-SVM", "RFE-SVM", "RF", "PermFIT-RF", "Vanilla-RF"),
           MSE = apply(pred, 2, function(x) mse(y = y, x)),
           Correlation = apply(pred, 2, function(x) cor(y, x)))
## Performance is evaluated via 5-fold CV, randomly repeated for 100 times

