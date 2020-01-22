###########################################################
### Blocked Feedforward Deep Neural Nets

#' Blocked Feedforward Deep Neural Nets
#'
#' Fit a Blocked Feedforward Deep Neural Network Model for Regression or Classification
#'
#' @param train A \code{dnnetInput} object, the training set.
#' @param validate A \code{dnnetInput} object, the validation set, optional.
#' @param n.hidden A list of numeric vectors for the blocked hidden structures, starting from the
#'  input layer.
#' @param norm.x A boolean variable indicating whether to normalize the input matrix.
#' @param norm.y A boolean variable indicating whether to normalize the response (if continuous).
#' @param activate Activation Function. One of the following,
#'  "sigmoid", "tanh", "relu", "prelu", "elu", "celu".
#' @param learning.rate Initial learning rate, 0.001 by default; If "adam" is chosen as
#'  an adaptive learning rate adjustment method, 0.1 by defalut.
#' @param l1.reg weight for l1 regularization, optional.
#' @param l2.reg weight for l2 regularization, optional.
#' @param n.batch Batch size for batch gradient descent.
#' @param n.epoch Maximum number of epochs.
#' @param early.stop Indicate whether early stop is used (only if there exists a validation set).
#' @param early.stop.det Number of epochs of increasing loss to determine the early stop.
#' @param plot Indicate whether to plot the loss.
#' @param accel "rcpp" to use the Rcpp version and "none" (default) to use the R version for back propagation.
#' @param learning.rate.adaptive Adaptive learning rate adjustment methods, one of the following,
#'  "constant", "adadelta", "adagrad", "momentum", "adam".
#' @param rho A parameter used in momentum.
#' @param epsilon A parameter used in Adagrad and Adam.
#' @param beta1 A parameter used in Adam.
#' @param beta2 A parameter used in Adam.
#' @param loss.f Loss function of choice.
#' @param load.param Whether initial parameters are loaded into the model.
#' @param initial.param The initial parameters to be loaded.
#'
#' @return Returns a \code{DnnModelObj} object.
#'
#' @importFrom stats runif
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' \code{\link{dnnetInput-class}}\cr
#' \code{\link{actF}}
#'
#' @export
dnnet_block <- function(train, validate = NULL,
                        load.param = FALSE, initial.param = NULL,
                        norm.x = TRUE, norm.y = ifelse(is.factor(train@y), FALSE, TRUE),
                        activate = "relu", n.hidden = list(dim(train@x)[2], 10, 10),
                        learning.rate = ifelse(learning.rate.adaptive %in% c("adam"), 0.001, 0.01),
                        l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 100,
                        early.stop = ifelse(is.null(validate), FALSE, TRUE), early.stop.det = 1000,
                        plot = FALSE, accel = c("rcpp", "none")[1],
                        learning.rate.adaptive = c("constant", "adadelta", "adagrad", "momentum", "adam")[5],
                        rho = c(0.9, 0.95, 0.99, 0.999)[ifelse(learning.rate.adaptive == "momentum", 1, 3)],
                        epsilon = c(10**-10, 10**-8, 10**-6, 10**-4)[2],
                        beta1 = 0.9, beta2 = 0.999, loss.f = ifelse(is.factor(train@y), "logit", "mse")) {

  if(!class(train@x) %in% c("matrix", "data.frame"))
    stop("x has to be either a matrix or a data frame. ")
  if(!class(train@y) %in% c("numeric", "factor", "vector", "integer"))
    stop("y has to be either a factor or a numeric vector. ")
  if(dim(train@x)[1] != length(train@y))
    stop("Dimensions of x and y do not match. ")

  if(is.null(validate))
    validate <- train

  learning.rate
  norm.y
  rho
  loss.f

  sample.size <- length(train@y)
  n.variable <- dim(train@x)[2]

  if(dim(train@x)[2] != sum(n.hidden[[1]]))
    stop("The dimension of the input must match the summation of numbers of nodes in the first layer. ")

  if(!is.null(train@w)) {

    if(length(train@w) != length(train@y))
      stop("Dimensions of y and w do not match. ")
    if(!class(train@w) %in% c("integer", "numeric"))
      stop("w has to be a numeric vector. ")
    if(sum(train@w < 0) > 0)
      stop("w has be to non-negative. ")
  } else {

    train@w <- rep(1, sample.size)
  }

  train@x <- as.matrix(train@x)
  if(load.param) {

    norm <- initial.param@norm
    train@x <- (train@x - outer(rep(1, dim(train@x)[1]), norm$x.center)) /
      outer(rep(1, dim(train@x)[1]), norm$x.scale)
    validate@x <- (validate@x - outer(rep(1, length(validate@y)), norm$x.center)) /
      outer(rep(1, length(validate@y)), norm$x.scale)

    if(is.factor(train@y)) {

      label <- levels(train@y)
      train@y <- (train@y == label[1])*1

      if(!is.null(validate))
        validate@y <- (validate@y == label[1])*1

      model.type <- "classification"
    } else {

      train@y <- (train@y - norm$y.center)/norm$y.scale
      validate@y <- (validate@y - norm$y.center)/norm$y.scale
      model.type <- "regression"
    }
  } else {

    norm <- list(x.center = rep(0, n.variable),
                 x.scale = rep(1, n.variable),
                 y.center = 0, y.scale = 1)
    if(norm.x && (sum(apply(train@x, 2, sd) == 0) == 0)) {

      train@x <- scale(train@x)
      norm$x.center <- attr(train@x, "scaled:center")
      norm$x.scale <- attr(train@x, "scaled:scale")

      if(!is.null(validate))
        validate@x <- (validate@x - outer(rep(1, length(validate@y)), norm$x.center)) /
        outer(rep(1, length(validate@y)), norm$x.scale)
    }

    if(is.factor(train@y)) {

      label <- levels(train@y)
      train@y <- (train@y == label[1])*1

      if(!is.null(validate))
        validate@y <- (validate@y == label[1])*1

      model.type <- "classification"
    } else {

      if(norm.y) {

        train@y <- scale(train@y)
        norm$y.center <- attr(train@y, "scaled:center")
        norm$y.scale <- attr(train@y, "scaled:scale")

        if(!is.null(validate))
          validate@y <- (validate@y - norm$y.center)/norm$y.scale
      }
      model.type <- "regression"
    }
  }

  if(sum(is.na(train@x)) > 0 | sum(is.na(train@y)) > 0)
    stop("Please remove NA's in the input data first. ")

  # w.ini.matrix <- initial.param
  w.ini <- 0.1
  if(load.param) {

    weight.ini <- initial.param@weight
    bias.ini <- initial.param@bias
  } else {

    weight.ini <- bias.ini <- list()
  }

  if(accel == "rcpp") {

    if(!is.null(validate)) {

      try(result <- backprop_BLOCK(n.hidden, w.ini, load.param, weight.ini, bias.ini,
                                   train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                                   activate,
                                   n.epoch, n.batch, model.type,
                                   learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                   learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    } else {

      try(result <- backprop_BLOCK(n.hidden, w.ini, load.param, weight.ini, bias.ini,
                                   train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                                   activate,
                                   n.epoch, n.batch, model.type,
                                   learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                   learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    }
  } else {

    if(!is.null(validate)) {
      try(result <- dnnet.backprop.r(n.hidden, w.ini, load.param, initial.param,
                                     train@x, train@y, train@w, TRUE, validate@x, validate@y, validate@w,
                                     get(activate), get(paste(activate, "_", sep = '')),
                                     n.epoch, n.batch, model.type,
                                     learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                     learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    } else {

      try(result <- dnnet.backprop.r(n.hidden, w.ini, load.param, initial.param,
                                     train@x, train@y, train@w, FALSE, matrix(0), matrix(0), matrix(0),
                                     get(activate), get(paste(activate, "_", sep = '')),
                                     n.epoch, n.batch, model.type,
                                     learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                     learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
    }
  }

  if(!is.null(validate) & plot) try(plot(result[[3]][0:result[[4]]+1]*norm$y.scale**2, ylab = "loss"))
  if(is.na(result[[3]][1]) | is.nan(result[[3]][1])) {

    min.loss <- Inf
  } else {

    if(early.stop) {

      min.loss <- min(result[[3]][0:result[[4]]+1])*norm$y.scale**2
    } else {

      min.loss <- result[[3]][length(result[[3]])]*norm$y.scale**2
    }
  }

  if(exists("result")) {

    return(methods::new("dnnet", norm = norm,
                        weight = result[[1]],
                        bias = result[[2]],
                        loss = min.loss,
                        loss.traj = as.numeric(result[[3]]*norm$y.scale**2),
                        label = ifelse(model.type == "regression", '', list(label))[[1]],
                        model.type = model.type,
                        model.spec = list(n.hidden = n.hidden,
                                          activate = activate,
                                          learning.rate = learning.rate,
                                          l1.reg = l1.reg,
                                          l2.reg = l2.reg,
                                          n.batch = n.batch,
                                          n.epoch = n.epoch)))
  } else {

    stop("Error fitting model. ")
  }
}
