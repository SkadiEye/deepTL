PermFIT
================

Permutation-based Feature Importance Test, a permutation-based feature
importance test scheme for black-box models (deep neural networks,
support vector machines, random forests, etc).

## Simulation setup

``` r
suppressMessages(library(deepTL))
library(MASS)
devtools::load_all()
```

    ## Loading deepTL

``` r
set.seed(1234)  
n <- 1000
p <- 20
n_test <- 10000

x <- mvrnorm(n, rep(0, p), diag(p))
y <- x[, 1] + log(1 + (x[, 3] + 1)**2 + x[, 2]**2*2)*2 + x[, 4]*x[, 5] + rnorm(n)

x_test <- mvrnorm(n_test, rep(0, p), diag(p))
y_test <- x_test[, 1] + log(1 + (x_test[, 3] + 1)**2 + x_test[, 2]**2*2)*2 + x_test[, 4]*x_test[, 5] + rnorm(n)
```

## Set hyper-parameters for DNN and PermFIT

``` r
esCtrl <- list(n.hidden = c(50, 40, 30, 20), activate = "relu",
               l1.reg = 10**-4, early.stop.det = 1000, n.batch = 50,
               n.epoch = 100, learning.rate.adaptive = "adam", plot = FALSE)
n_ensemble <- 10
n_perm <- 10
dnn_obj <- importDnnet(x = x, y = y)
dat_spl <- splitDnnet(dnn_obj, 0.8)
```

## ***PermFIT-DNN***

``` r
permfit_dnn <- permfit(train = dat_spl$train, validate = dat_spl$valid, k_fold = 0,
                       n_perm = n_perm, method = "ensemble_dnnet", 
                       n.ensemble = n_ensemble, esCtrl = esCtrl, verbose = FALSE)
## Importance for first 10 features
permfit_dnn@importance[1:10, c("importance", "importance_pval")]
```

    ##     importance importance_pval
    ## 1   2.02984975    0.000000e+00
    ## 2   1.82662310    0.000000e+00
    ## 3   1.78360477    3.210787e-11
    ## 4   1.43118761    5.946830e-09
    ## 5   1.33929423    1.759480e-08
    ## 6  -0.02743449    8.747284e-01
    ## 7  -0.04486253    9.532483e-01
    ## 8   0.05302336    5.459916e-02
    ## 9  -0.01921513    8.292609e-01
    ## 10 -0.01905376    7.504268e-01

## Performance improvement with PermFIT-DNN

``` r
dnn_obj <- importDnnet(x = x, y = y)
dnn_mod <- ensemble_dnnet(dnn_obj, n_ensemble, esCtrl, verbose = FALSE)
dnn_pred <- predict(dnn_mod, x_test)

# selected features with p-values < 0.1 
imp_feature <- which(permfit_dnn@importance$importance_pval < 0.1)
dnn_obj0 <- importDnnet(x = x[, imp_feature], y = y)
dnn_mod0 <- ensemble_dnnet(dnn_obj0, n_ensemble, esCtrl, verbose = FALSE)
dnn_pred0 <- predict(dnn_mod0, x_test[, imp_feature])

data.frame(method = c("DNN", "PermFIT-DNN"), 
           mse = c(mean((dnn_pred - y_test)**2), mean((dnn_pred0 - y_test)**2)), 
           cor = c(cor(dnn_pred, y_test), cor(dnn_pred0, y_test)))
```

    ##        method      mse       cor
    ## 1         DNN 1.466915 0.8479872
    ## 2 PermFIT-DNN 1.279319 0.8688133

## **PermFIT-SVM**

``` r
permfit_svm <- permfit(train = dat_spl$train, validate = dat_spl$valid, k_fold = 0,
                       n_perm = n_perm, method = "svm")
## Importance for first 10 features
permfit_svm@importance[1:10, c("importance", "importance_pval")]
```

    ##      importance importance_pval
    ## 1   1.705174874    2.220446e-16
    ## 2   1.240645805    2.109424e-15
    ## 3   1.708539879    1.487699e-14
    ## 4   1.252229598    3.315890e-07
    ## 5   1.309370661    3.812333e-05
    ## 6  -0.051113292    9.240859e-01
    ## 7  -0.063150100    8.776631e-01
    ## 8  -0.004447905    5.370607e-01
    ## 9  -0.034412107    9.399677e-01
    ## 10 -0.025787071    7.387451e-01

## Performance improvement with PermFIT-SVM

``` r
library(e1071)
```

    ## 
    ## Attaching package: 'e1071'

    ## The following object is masked from 'package:deepTL':
    ## 
    ##     sigmoid

``` r
svm_mod <- tune.svm(x, y, gamma = 10**(-(0:4)), cost = 10**(0:4/2), tunecontrol = tune.control(cross = 5))
svm_pred <- predict(svm_mod$best.model, x_test)

imp_feature <- which(permfit_svm@importance$importance_pval < 0.1)
svm_mod0 <- tune.svm(x[, imp_feature], y, gamma = 10**(-(0:4)), cost = 10**(0:4/2), tunecontrol = tune.control(cross = 5))
svm_pred0 <- predict(svm_mod0$best.model, x_test[, imp_feature])

data.frame(method = c("SVM", "PermFIT-SVM"), 
           mse = c(mean((svm_pred - y_test)**2), mean((svm_pred0 - y_test)**2)), 
           cor = c(cor(svm_pred, y_test), cor(svm_pred0, y_test)))
```

    ##        method      mse       cor
    ## 1         SVM 1.797506 0.8096656
    ## 2 PermFIT-SVM 1.341164 0.8616215
