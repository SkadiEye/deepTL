dnnet
================

Basic feed-forward, fully-connected neural networks, and bootstrap
aggregating (bagging).

## A simulation for continuous outcome with Gaussian error

``` r
suppressMessages(library(deepTL))
library(MASS)
set.seed(1234)  
n <- 500
p <- 20
n_test <- 10000

x <- mvrnorm(n, rep(0, p), diag(p))
y <- x[, 1] + log(1 + (x[, 3] + 1)**2 + x[, 2]**2*2)*2 + x[, 4]*x[, 5] + rnorm(n)

x_test <- mvrnorm(n_test, rep(0, p), diag(p))
y_test <- x_test[, 1] + log(1 + (x_test[, 3] + 1)**2 + x_test[, 2]**2*2)*2 + x_test[, 4]*x_test[, 5] + rnorm(n)
```

## Hyper-parameters

``` r
esCtrl <- list(n.hidden = c(50, 40, 30, 20), activate = "relu",
               l1.reg = 10**-4, early.stop.det = 1000, n.batch = 50,
               n.epoch = 100, learning.rate.adaptive = "adam", plot = FALSE)
dnn_obj <- importDnnet(x = x, y = y)
dat_spl <- splitDnnet(dnn_obj, 0.8)
```

## dnnet

``` r
dnn_mod <- do.call("dnnet", c(list("train" = dat_spl$train, "validate" = dat_spl$valid), esCtrl))
dnn_pred <- predict(dnn_mod, x_test)
```

## Bootstrap aggregating (Bagging)

``` r
dnn_en_mod <- ensemble_dnnet(dnn_obj, 10, esCtrl, verbose = FALSE)
dnn_en_pred <- predict(dnn_en_mod, x_test)
```

## More DNNs in Bagging

``` r
dnn_en_100_mod <- ensemble_dnnet(dnn_obj, 100, esCtrl, best.opti = FALSE, verbose = FALSE)
dnn_en_100_pred <- predict(dnn_en_100_mod, x_test)
```

## Bootstrap aggregating (Bagging) with optimal subset of DNNs

``` r
dnn_en_opt_mod <- ensemble_dnnet(dnn_obj, 100, esCtrl, verbose = FALSE)
dnn_en_opt_pred <- predict(dnn_en_opt_mod, x_test)
```

## Performances

``` r
data.frame(dnn = c("A single DNN", "Bagging of 10 DNNs", "Bagging of 100 DNNs", "Bagging of 100 DNNs with optimal subset"), 
           mse = c(mean((dnn_pred - y_test)**2), mean((dnn_en_pred - y_test)**2), 
                   mean((dnn_en_100_pred - y_test)**2), mean((dnn_en_opt_pred - y_test)**2)))
```

    ##                                       dnn      mse
    ## 1                            A single DNN 2.395480
    ## 2                      Bagging of 10 DNNs 2.037568
    ## 3                     Bagging of 100 DNNs 1.913064
    ## 4 Bagging of 100 DNNs with optimal subset 1.834117
