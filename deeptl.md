deepTL
================

Deep Treatment Learning, an efficient semiparametric framework coupled
with ensemble DNNs for adjusting complex confounding. (Paper submitted)

## Simulation setup

``` r
suppressMessages(library(deepTL))
library(MASS)
set.seed(1234)  
n <- 5000
p <- 20
beta.true <- 1
sigma <- 1
alpha1 <- runif(5)*2 - 1
alpha2 <- runif(5)*2 - 1

x <- mvrnorm(n, rep(0, p), diag(p))
prob <- 1/(1 + exp(1 - x[, 1:5] %*% alpha1 - 2 * cos(x[, 8])))
z <- rbinom(n, 1, prob)
y <- -1 + beta.true * z + as.numeric(x[, 1:5] %*% alpha2) - cos(x[, 8] * 2) + x[, 9]**2 + rnorm(n) * sigma
```

## Hyper-parameters for DNN and ensemble

``` r
#### Hyper-parameters
n_ensemble <- 10
esCtrl1 <- list(n.hidden = c(50, 40, 30, 20),
                l1.reg = 10**-4, n.batch = 50, n.epoch = 100, early.stop.det = 1000,
                accel = "rcpp", learning.rate.adaptive = "adam", plot = FALSE)
en_dnn_ctrl <- list(esCtrl = esCtrl1, n.ensemble = n_ensemble, verbose = 0)
```

## deepTL and semiDNN

``` r
#### deepTL/semiDNN
set.seed(4321)
trt_obj <- importTrt(x = x, y = y, z = z)
double_deepTL(trt_obj, en_dnn_ctrl, en_dnn_ctrl, methods = c("revised-semi", "semi"))
```

    ## Fitting E(Z|X) ... 
    ## Fitting E(Y - b1*Z|X) ... 
    ## Fitting E(Y|X) ...

    ##                     method      beta         var
    ## z_nmrc revised-semi-dnn-en 0.9891637 0.001112008
    ## 1              semi-dnn-en 0.9648760 0.001139422

# References
