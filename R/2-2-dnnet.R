###########################################################
### Multilayer Perceptron Model for Regression or Classification

#' Multilayer Perceptron Model for Regression or Classification
#'
#' Fit a Multilayer Perceptron Model for Regression or Classification
#'
#' @param train A \code{dnnetInput} or a \code{dnnetSurvInput} object, the training set.
#' @param validate A \code{dnnetInput} or a \code{dnnetSurvInput} object, the validation set, optional.
#' @param family If it's specified as Poisson, a poisson model with log link will be fitted.
#' @param norm.x A boolean variable indicating whether to normalize the input matrix.
#' @param norm.y A boolean variable indicating whether to normalize the response (if continuous).
#' @param n.hidden A numeric vector for numbers of nodes for all hidden layers.
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
#'  "constant", "adam", "amsgrad", "adadelta", "adagrad".
#' @param rho A parameter used in momentum.
#' @param epsilon A parameter used in Adagrad and Adam.
#' @param beta1 A parameter used in Adam.
#' @param beta2 A parameter used in Adam.
#' @param loss.f Loss function of choice.
#' @param pathway If pathway.
#' @param pathway.list Pathway list.
#' @param pathway.active Activation function for the pathway layer.
#' @param l1.pathway The weight for l1-panelty for the pathway layer.
#' @param l2.pathway The weight for l2-panelty for the pathway layer.
#' @param load.param Whether initial parameters are loaded into the model.
#' @param initial.param The initial parameters to be loaded.
#'
#' @return Returns a \code{DnnModelObj} object.
#'
#' @importFrom stats runif
#' @importFrom stats sd
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' \code{\link{dnnetInput-class}}\cr
#' \code{\link{actF}}
#'
#' @export
dnnet <- function(train, validate = NULL,
                  load.param = FALSE, initial.param = NULL,
                  family = c("gaussin", "binomial", "multinomial", "poisson", "negbin",
                             "coxph", "poisson-nonzero", "negbin-nonzero", "zip", "zinb")[1],
                  norm.x = TRUE,
                  norm.y = ifelse(is.factor(train@y) || (class(train) == "dnnetSurvInput") ||
                                    (family %in% c("poisson", "poisson-nonzero", "zip", "zinb")), FALSE, TRUE),
                  activate = "relu", n.hidden = c(10, 10),
                  learning.rate = ifelse(learning.rate.adaptive %in% c("adam"), 0.001, 0.01),
                  l1.reg = 0, l2.reg = 0, n.batch = 100, n.epoch = 100,
                  early.stop = ifelse(is.null(validate), FALSE, TRUE), early.stop.det = 1000,
                  plot = FALSE, accel = c("rcpp", "none")[1],
                  learning.rate.adaptive = c("constant", "adadelta", "adagrad", "momentum", "adam", "amsgrad")[5],
                  rho = c(0.9, 0.95, 0.99, 0.999)[ifelse(learning.rate.adaptive == "momentum", 1, 3)],
                  epsilon = c(10**-10, 10**-8, 10**-6, 10**-4)[2],
                  beta1 = 0.9, beta2 = 0.999, loss.f = ifelse(is.factor(train@y), "logit", "mse"),
                  pathway = FALSE, pathway.list = NULL, pathway.active = "identity", l1.pathway = 0, l2.pathway = 0) {

  if(!class(train@x) %in% c("matrix", "data.frame"))
    stop("x has to be either a matrix or a data frame. ")
  if(!class(train@y)[1] %in% c("numeric", "factor", "ordered", "vector", "integer", "matrix"))
    stop("y has to be either a factor, a numeric vector, or a matrix. ")
  if(class(train@y)[1] != "matrix") {
    if(dim(train@x)[1] != length(train@y))
      stop("Dimensions of x and y do not match. ")
  } else {
    if(dim(train@x)[1] != dim(train@y)[1])
      stop("Dimensions of x and y do not match. ")
  }

  if(is.null(validate))
    validate <- train

  learning.rate
  norm.y
  rho
  loss.f

  sample.size <- dim(train@x)[1]
  if(!is.null(validate))
    valid.size <- dim(validate@x)[1]
  n.variable <- dim(train@x)[2]
  if(class(train@y)[1] != "matrix")
    n.outcome <- 1
  else
    n.outcome <- dim(train@y)[2]

  if(n.outcome == 1 && (!is.factor(train@y)))
    train@y <- c(train@y)

  if(!is.null(train@w)) {

    if(length(train@w) != sample.size)
      stop("Dimensions of x and w do not match. ")
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
    train@x <- (train@x - outer(rep(1, sample.size), norm$x.center)) /
      outer(rep(1, sample.size), norm$x.scale)
    if(!is.null(validate))
      validate@x <- (validate@x - outer(rep(1, valid.size), norm$x.center)) /
      outer(rep(1, valid.size), norm$x.scale)

    if(is.factor(train@y)) {

      label <- levels(train@y)
      if(length(label) == 2) {

        train@y <- (train@y == label[1])*1
        if(!is.null(validate))
          validate@y <- (validate@y == label[1])*1

        model.type <- "binary-classification"
      } else {

        if(!is.ordered(train@y)) {

          dim_y <- length(label)
          mat_y <- matrix(0, sample.size, dim_y)
          if(!is.null(validate))
            mat_y_valid <- matrix(0, length(validate@y), dim_y)
          for(d in 1:dim_y) {
            mat_y[, d] <- (train@y == label[d])*1
            if(!is.null(validate))
              mat_y_valid[, d] <- (validate@y == label[d])*1
          }
          train@y <- mat_y
          if(!is.null(validate))
            validate@y <- mat_y_valid

          model.type <- "multi-classification"
          if(loss.f == "logit") loss.f <- "cross-entropy"
        } else {

          dim_y <- length(label)
          mat_y <- matrix(0, sample.size, dim_y)
          if(!is.null(validate))
            mat_y_valid <- matrix(0, length(validate@y), dim_y)
          for(d in 1:sample.size)
            mat_y[d, 1:match(train@y[d], label)] <- 1
          for(d in 1:valid.size)
            if(!is.null(validate))
              mat_y_valid[d, 1:match(validate@y[d], label)] <- 1

          train@y <- mat_y[, -1]
          if(!is.null(validate))
            validate@y <- mat_y_valid[, -1]

          model.type <- "ordinal-multi-classification"
          # if(loss.f == "logit") loss.f <- "cross-entropy"
        }
      }
    } else if(n.outcome == 1) {

      train@y <- (train@y - norm$y.center)/norm$y.scale
      if(!is.null(validate))
        validate@y <- (validate@y - norm$y.center)/norm$y.scale
      model.type <- "regression"
    } else {

      train@y <- (train@y - rep(1, sample.size) %*% t(norm$y.center))/
        (rep(1, sample.size) %*% t(norm$y.scale))
      if(!is.null(validate))
        validate@y <- (validate@y - rep(1, sample.size) %*% t(norm$y.center))/
          (rep(1, sample.size) %*% t(norm$y.scale))
      model.type <- "multi-regression"
    }
  } else {

    norm <- list(x.center = rep(0, n.variable),
                 x.scale = rep(1, n.variable),
                 y.center = rep(0, n.outcome),
                 y.scale = rep(1, n.outcome))
    if(norm.x && (sum(apply(train@x, 2, stats::sd) == 0) == 0)) {

      train@x <- scale(train@x)
      norm$x.center <- attr(train@x, "scaled:center")
      norm$x.scale <- attr(train@x, "scaled:scale")

      if(!is.null(validate))
        validate@x <- (validate@x - outer(rep(1, valid.size), norm$x.center))/
        outer(rep(1, valid.size), norm$x.scale)
    }

    if(is.factor(train@y)) {


      label <- levels(train@y)
      if(length(label) == 2) {

        train@y <- (train@y == label[1])*1
        if(!is.null(validate))
          validate@y <- (validate@y == label[1])*1

        model.type <- "binary-classification"
      } else {

        if(!is.ordered(train@y)) {

          dim_y <- length(label)
          mat_y <- matrix(0, sample.size, dim_y)
          if(!is.null(validate))
            mat_y_valid <- matrix(0, length(validate@y), dim_y)
          for(d in 1:dim_y) {
            mat_y[, d] <- (train@y == label[d])*1
            if(!is.null(validate))
              mat_y_valid[, d] <- (validate@y == label[d])*1
          }
          train@y <- mat_y
          if(!is.null(validate))
            validate@y <- mat_y_valid

          model.type <- "multi-classification"
          if(loss.f == "logit") loss.f <- "cross-entropy"
        } else {

          dim_y <- length(label)
          mat_y <- matrix(0, sample.size, dim_y)
          if(!is.null(validate))
            mat_y_valid <- matrix(0, length(validate@y), dim_y)
          for(d in 1:length(train@y))
            mat_y[d, 1:match(train@y[d], label)] <- 1
          for(d in 1:length(validate@y))
            if(!is.null(validate))
              mat_y_valid[d, 1:match(validate@y[d], label)] <- 1

          train@y <- mat_y[, -1]
          if(!is.null(validate))
            validate@y <- mat_y_valid[, -1]

          model.type <- "ordinal-multi-classification"
        }
      }
    } else {

      if(norm.y && (!family %in% c("poisson", "negbin", "poisson-nonzero",
                                   "negbin-nonzero", "zip", "zinb"))) {

        train@y <- scale(train@y)
        norm$y.center <- attr(train@y, "scaled:center")
        norm$y.scale <- attr(train@y, "scaled:scale")

        if(!is.null(validate))
          if(class(train@y)[1] != "matrix")
            validate@y <- (validate@y - norm$y.center)/norm$y.scale
        if(!is.null(validate))
          if(class(train@y)[1] == "matrix")
            validate@y <- (validate@y - rep(1, valid.size) %*% t(norm$y.center))/
          (rep(1, valid.size) %*% t(norm$y.scale))
      }
      if(n.outcome == 1) {
        model.type <- "regression"
      } else {
        model.type <- "multi-regression"
      }
    }
  }

  if(class(train) == "dnnetSurvInput")
    model.type <- "survival"

  if(family %in% c("poisson", "poisson-nonzero")) {
    model.type <- "poisson"
    loss.f <- family
  }

  if(family %in% c("negbin", "negbin-nonzero")) {
    model.type <- "negbin"
    loss.f <- family
  }

  if(family %in% c("zip", "zinb")) {
    model.type <- family
    loss.f <- family
    train@y <- cbind((train@y > 0)*1, train@y)
    if(!is.null(validate))
      validate@y <- cbind((validate@y > 0)*1, validate@y)
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

  # print(head(as.matrix(train@y)))

  if(pathway) {

    x_pathway <- list()
    if(!is.null(validate))
      x_valid_pathway <- list()
    for(i in 1:length(pathway.list)) {
      x_pathway[[i]] <- train@x[, pathway.list[[i]]]
      if(!is.null(validate))
        x_valid_pathway[[i]] <- validate@x[, pathway.list[[i]]]
    }
    x_input <- cbind(matrix(0, sample.size, length(pathway.list)), train@x[, -unlist(pathway.list)])
    if(!is.null(validate))
      x_valid_input <- cbind(matrix(0, dim(validate@x)[1], length(pathway.list)), validate@x[, -unlist(pathway.list)])

    weight_pathway.ini <- list()
    bias_pathway.ini <- 0
    if(load.param) {

      weight_pathway.ini <- initial.param@model.spec$weight.pathway
      bias_pathway.ini <- initial.param@model.spec$bias.pathway
    }
  }

  # browser()
  if(accel == "rcpp") {

    if(pathway) {
      if(!is.null(validate)) {

        try(result <- backprop_long(n.hidden, w.ini,
                                    weight_pathway.ini, bias_pathway.ini,
                                    l1.pathway, l2.pathway,
                                    load.param, weight.ini, bias.ini,
                                    x_input, as.matrix(train@y), train@w, x_pathway, TRUE,
                                    x_valid_input, as.matrix(validate@y), validate@w, x_valid_pathway,
                                    activate, pathway.active,
                                    n.epoch, n.batch, model.type,
                                    learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                    learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
      } else {

        try(result <- backprop_long(n.hidden, w.ini,
                                    weight_pathway.ini, bias_pathway.ini,
                                    l1.pathway, l2.pathway,
                                    load.param, weight.ini, bias.ini,
                                    x_input, as.matrix(train@y), train@w, x_pathway, FALSE,
                                    matrix(0), matrix(0), matrix(0), list(),
                                    activate, pathway.active,
                                    n.epoch, n.batch, model.type,
                                    learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                    learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
      }
    } else if(class(train) == "dnnetSurvInput") {

      if(!is.null(validate)) {

        try(result <- backprop_surv(n.hidden, w.ini, load.param, weight.ini, bias.ini,
                                    train@x, cbind(train@y, train@e), train@w, TRUE,
                                    validate@x, cbind(validate@y, validate@e), validate@w,
                                    activate,
                                    n.epoch, n.batch, model.type,
                                    learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                    learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
      } else {

        try(result <- backprop_surv(n.hidden, w.ini, load.param, weight.ini, bias.ini,
                                    train@x, cbind(train@y, train@e), train@w, FALSE, matrix(0), matrix(0), matrix(0),
                                    activate,
                                    n.epoch, n.batch, model.type,
                                    learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                                    learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
      }
    } else {

      if(!is.null(validate)) {

        try(result <- backprop(n.hidden, w.ini, load.param, weight.ini, bias.ini,
                               train@x, as.matrix(train@y), train@w, TRUE, validate@x, as.matrix(validate@y), validate@w,
                               activate,
                               n.epoch, n.batch, model.type,
                               learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                               learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
      } else {

        try(result <- backprop(n.hidden, w.ini, load.param, weight.ini, bias.ini,
                               train@x, as.matrix(train@y), train@w, FALSE, matrix(0), matrix(0), matrix(0),
                               activate,
                               n.epoch, n.batch, model.type,
                               learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                               learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f))
      }
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

  if(!is.null(validate) & plot)
    try(plot(result[[3]][0:result[[4]]+1]*mean(norm$y.scale**2), ylab = "loss"))
  if(is.na(result[[3]][1]) | is.nan(result[[3]][1])) {

    min.loss <- Inf
  } else {

    if(early.stop) {

      min.loss <- min(result[[3]][0:result[[4]]+1])*mean(norm$y.scale**2)
    } else {

      min.loss <- result[[3]][length(result[[3]])]*mean(norm$y.scale**2)
    }
  }

  weight.pathway <- bias.pathway <- NULL
  if(pathway) {
    weight.pathway <- result[[5]]
    bias.pathway <- result[[6]]
  }

  if(exists("result")) {

    return(methods::new("dnnet", norm = norm,
                        weight = result[[1]],
                        bias = result[[2]],
                        loss = min.loss,
                        loss.traj = as.numeric(result[[3]]*mean(norm$y.scale**2)),
                        label = ifelse(model.type %in% c("multi-regression", "regression", "survival",
                                                         "poisson", "negbin", "zip", "zinb"),
                                       '', list(label))[[1]],
                        model.type = model.type,
                        model.spec = list(n.hidden = n.hidden,
                                          activate = activate,
                                          learning.rate = learning.rate,
                                          l1.reg = l1.reg,
                                          l2.reg = l2.reg,
                                          n.batch = n.batch,
                                          n.epoch = n.epoch,
                                          pathway = pathway,
                                          pathway.list = pathway.list,
                                          pathway.active = pathway.active,
                                          l1.pathway = l1.pathway,
                                          l2.pathway = l2.pathway,
                                          weight.pathway = weight.pathway,
                                          bias.pathway = bias.pathway,
                                          negbin_alpha = ifelse(model.type %in% c("negbin", "zinb"),
                                                                result[[5]], 1))))
  } else {

    stop("Error fitting model. ")
  }
}

#' Back Propagation
#'
#' @param n.hidden A numeric vector for numbers of nodes for all hidden layers.
#' @param w.ini Initial weight parameter.
#' @param load.param Whether initial parameters are loaded into the model.
#' @param initial.param The initial parameters to be loaded.
#' @param x x
#' @param y y
#' @param w w
#' @param valid If exists the validation set
#' @param x.valid x-valid
#' @param y.valid y-valid
#' @param w.valid w-valid
#' @param activate Activation Function.
#' @param activate_ The forst derivative of the activation function.
#' @param n.batch Batch size for batch gradient descent.
#' @param n.epoch Maximum number of epochs.
#' @param model.type Type of model.
#' @param learning.rate Initial learning rate, 0.001 by default; If "adam" is chosen as
#'  an adaptive learning rate adjustment method, 0.1 by defalut.
#' @param l1.reg weight for l1 regularization, optional.
#' @param l2.reg weight for l2 regularization, optional.
#' @param early.stop Indicate whether early stop is used (only if there exists a validation set).
#' @param early.stop.det Number of epochs of increasing loss to determine the early stop.
#' @param learning.rate.adaptive Adaptive learning rate adjustment methods, one of the following,
#'  "constant", "adadelta", "adagrad", "momentum", "adam".
#' @param rho A parameter used in momentum.
#' @param epsilon A parameter used in Adagrad and Adam.
#' @param beta1 A parameter used in Adam.
#' @param beta2 A parameter used in Adam.
#' @param loss.f Loss function of choice.
#'
#' @return Returns a \code{list} of results to \code{dnnet}.
dnnet.backprop.r <- function(n.hidden, w.ini, load.param, initial.param,
                             x, y, w, valid, x.valid, y.valid, w.valid,
                             activate, activate_, n.epoch, n.batch, model.type,
                             learning.rate, l1.reg, l2.reg, early.stop, early.stop.det,
                             learning.rate.adaptive, rho, epsilon, beta1, beta2, loss.f) {

  if(valid) {

    valid_sample_size <- matrix(rep(1, nrow(x.valid)), nrow(x.valid), 1)
  }

  loss <- numeric(0)
  weight <- list()
  bias <- list()
  a <- list()
  h <- list()
  d_a <- list()
  d_h <- list()
  d_w <- list()

  best.loss <- Inf
  best.weight <- list()
  best.bias <- list()

  n.layer <- length(n.hidden)
  n.variable <- ncol(x)
  sample.size <- nrow(x)

  if(load.param) {

    weight <- initial.param@weight
    bias <- initial.param@bias
  } else {

    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        weight[[i]] <- matrix(stats::runif(n.variable * n.hidden[i], -1, 1), n.variable, n.hidden[i]) * w.ini
        bias[[i]]   <- matrix(stats::runif(n.hidden[i], -1, 1), 1, n.hidden[i]) * w.ini / 2
      } else if(i == n.layer + 1) {

        weight[[i]] <- matrix(stats::runif(n.hidden[i-1] * 1, -1, 1), n.hidden[i-1], 1) * w.ini
        bias[[i]]   <- matrix(stats::runif(1, -1, 1), 1, 1) * w.ini / 2
      } else {

        weight[[i]] <- matrix(stats::runif(n.hidden[i-1] * n.hidden[i], -1, 1), n.hidden[i-1], n.hidden[i]) * w.ini
        bias[[i]]   <- matrix(stats::runif(n.hidden[i], -1, 1), 1, n.hidden[i]) * w.ini / 2
      }
    }
  }

  best.weight <- weight
  best.bias <- bias

  dw <- list()
  db <- list()

  if(learning.rate.adaptive == "momentum") {

    last.dw <- list()
    last.db <- list()
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        last.dw[[i]] <- matrix(0, n.variable, n.hidden[i])
        last.db[[i]] <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        last.dw[[i]] <- matrix(0, n.hidden[i-1], 1)
        last.db[[i]] <- matrix(0, 1, 1)
      } else {

        last.dw[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        last.db[[i]] <- matrix(0, 1, n.hidden[i])
      }
    }
  } else if(learning.rate.adaptive == "adagrad") {

    weight.ss <- list()
    bias.ss <- list()
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        weight.ss[[i]] <- matrix(0, n.variable, n.hidden[i])
        bias.ss[[i]]   <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        weight.ss[[i]] <- matrix(0, n.hidden[i-1], 1)
        bias.ss[[i]]   <- matrix(0, 1, 1)
      } else {

        weight.ss[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        bias.ss[[i]]   <- matrix(0, 1, n.hidden[i])
      }
    }
  } else if(learning.rate.adaptive == "adadelta") {

    weight.egs <- list()
    weight.es <- list()
    bias.egs <- list()
    bias.es <- list()
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        weight.egs[[i]] <- matrix(0, n.variable, n.hidden[i])
        bias.egs[[i]]   <- matrix(0, 1, n.hidden[i])
        weight.es[[i]]  <- matrix(0, n.variable, n.hidden[i])
        bias.es[[i]]    <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        weight.egs[[i]] <- matrix(0, n.hidden[i-1], 1)
        bias.egs[[i]]   <- matrix(0, 1, 1)
        weight.es[[i]]  <- matrix(0, n.hidden[i-1], 1)
        bias.es[[i]]    <- matrix(0, 1, 1)
      } else {

        weight.egs[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        bias.egs[[i]]   <- matrix(0, 1, n.hidden[i])
        weight.es[[i]]  <- matrix(0, n.hidden[i-1], n.hidden[i])
        bias.es[[i]]    <- matrix(0, 1, n.hidden[i])
      }
    }
  } else if(learning.rate.adaptive == "adam") {

    mt.w <- list()
    vt.w <- list()
    mt.b <- list()
    vt.b <- list()
    mt.ind <- 0
    for(i in 1:(n.layer + 1)) {

      if(i == 1) {

        mt.w[[i]] <- matrix(0, n.variable, n.hidden[i])
        mt.b[[i]] <- matrix(0, 1, n.hidden[i])
        vt.w[[i]] <- matrix(0, n.variable, n.hidden[i])
        vt.b[[i]] <- matrix(0, 1, n.hidden[i])
      } else if(i == n.layer + 1) {

        mt.w[[i]] <- matrix(0, n.hidden[i-1], 1)
        mt.b[[i]] <- matrix(0, 1, 1)
        vt.w[[i]] <- matrix(0, n.hidden[i-1], 1)
        vt.b[[i]] <- matrix(0, 1, 1)
      } else {

        mt.w[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        mt.b[[i]] <- matrix(0, 1, n.hidden[i])
        vt.w[[i]] <- matrix(0, n.hidden[i-1], n.hidden[i])
        vt.b[[i]] <- matrix(0, 1, n.hidden[i])
      }
    }
  }

  n.round <- ceiling(sample.size / n.batch)
  i.bgn <- integer(n.round)
  i.end <- integer(n.round)
  n.s <- integer(n.round)
  one_sample_size <- list()
  # one_sample_size_t <- list()
  for(s in 1:n.round) {

    i.bgn[s] <- (s-1)*n.batch + 1
    i.end[s] <- min(s*n.batch, sample.size)
    n.s[s] <- i.end[s] - i.bgn[s] + 1
    one_sample_size[[s]] <- matrix(rep(1, n.s[s]), n.s[s], 1)
    # one_sample_size_t[[s]] <- gpuR::t(one_sample_size[[s]])
  }

  for(k in 1:n.epoch) {

    new.order <- sample(sample.size)
    x_ <- x[new.order, ]
    y_ <- y[new.order]
    w_ <- w[new.order]

    for(i in 1:n.round) {

      xi <- x_[i.bgn[i]:i.end[i], ]
      yi <- y_[i.bgn[i]:i.end[i]]
      wi <- w_[i.bgn[i]:i.end[i]]

      for(j in 1:n.layer) {

        if(j == 1) {

          a[[j]] <- (xi %*% weight[[j]]) + one_sample_size[[i]] %*% bias[[j]]
          h[[j]] <- activate(a[[j]])
        } else {

          a[[j]] <- h[[j-1]] %*% weight[[j]] + one_sample_size[[i]] %*% bias[[j]]
          h[[j]] <- activate(a[[j]])
        }
      }

      y.pi <- h[[n.layer]] %*% weight[[n.layer + 1]] + one_sample_size[[i]] %*% bias[[n.layer + 1]]
      # if(model.type == "classification")
      if(loss.f == "logit")
        y.pi <- 1/(1 + exp(-y.pi))
      if(loss.f == "rmsle") {
        y.pi <- relu(y.pi)
        d_a[[n.layer + 1]] <- -(log(yi+1) - log(y.pi+1)) / (y.pi+1) * (y.pi > 0) * wi / sum(wi)
      } else {
        d_a[[n.layer + 1]] <- -(yi - y.pi) * wi / sum(wi)
      }

      d_w[[n.layer + 1]] <- t(h[[n.layer]]) %*% d_a[[n.layer + 1]]
      bias.grad <- (t(d_a[[n.layer + 1]]) %*% one_sample_size[[i]])
      if(learning.rate.adaptive == "momentum") {

        last.dw[[n.layer + 1]] <- last.dw[[n.layer + 1]] * rho + d_w[[n.layer + 1]] * learning.rate
        last.db[[n.layer + 1]] <- last.db[[n.layer + 1]] * rho + bias.grad * learning.rate
        dw[[n.layer + 1]] <- last.dw[[n.layer + 1]]
        db[[n.layer + 1]] <- last.db[[n.layer + 1]]
      } else if (learning.rate.adaptive == "adagrad") {

        weight.ss[[n.layer + 1]] <- weight.ss[[n.layer + 1]] + d_w[[n.layer + 1]]**2
        bias.ss[[n.layer + 1]]   <- bias.ss[[n.layer + 1]]   + bias.grad**2
        dw[[n.layer + 1]] <- d_w[[n.layer + 1]]/sqrt(weight.ss[[n.layer + 1]] + epsilon) * learning.rate
        db[[n.layer + 1]] <- bias.grad/sqrt(bias.ss[[n.layer + 1]] + epsilon) * learning.rate
      } else if (learning.rate.adaptive == "adadelta") {

        weight.egs[[n.layer + 1]] <- weight.egs[[n.layer + 1]] * rho + (1-rho) * d_w[[n.layer + 1]]**2
        bias.egs[[n.layer + 1]]   <-   bias.egs[[n.layer + 1]] * rho + (1-rho) * bias.grad**2
        dw[[n.layer + 1]] <- sqrt(weight.es[[n.layer + 1]] + epsilon) /
          sqrt(weight.egs[[n.layer + 1]] + epsilon) * d_w[[n.layer + 1]]
        db[[n.layer + 1]] <- sqrt(bias.es[[n.layer + 1]] + epsilon) /
          sqrt(bias.egs[[n.layer + 1]] + epsilon) * bias.grad
        weight.es[[n.layer + 1]] <- weight.es[[n.layer + 1]] * rho + (1-rho) * dw[[n.layer + 1]]**2
        bias.es[[n.layer + 1]]   <-   bias.es[[n.layer + 1]] * rho + (1-rho) * db[[n.layer + 1]]**2
      } else if (learning.rate.adaptive == "adam") {

        mt.ind <- mt.ind + 1
        mt.w[[n.layer + 1]] <- mt.w[[n.layer + 1]] * beta1 + (1-beta1) * d_w[[n.layer + 1]]
        mt.b[[n.layer + 1]] <- mt.b[[n.layer + 1]] * beta1 + (1-beta1) * bias.grad
        vt.w[[n.layer + 1]] <- vt.w[[n.layer + 1]] * beta2 + (1-beta2) * d_w[[n.layer + 1]]**2
        vt.b[[n.layer + 1]] <- vt.b[[n.layer + 1]] * beta2 + (1-beta2) * bias.grad**2
        dw[[n.layer + 1]] <- learning.rate * mt.w[[n.layer + 1]] / (1-beta1**mt.ind) /
          (sqrt(vt.w[[n.layer + 1]] / (1-beta2**mt.ind)) + epsilon)
        db[[n.layer + 1]] <- learning.rate * mt.b[[n.layer + 1]] / (1-beta1**mt.ind) /
          (sqrt(vt.b[[n.layer + 1]] / (1-beta2**mt.ind)) + epsilon)
      } else {

        dw[[n.layer + 1]] <- d_w[[n.layer + 1]] * learning.rate
        db[[n.layer + 1]] <- bias.grad * learning.rate
      }
      weight[[n.layer + 1]] <- weight[[n.layer + 1]] - dw[[n.layer + 1]] -
        l1.reg*((weight[[n.layer + 1]] > 0) - (weight[[n.layer + 1]] < 0)) -
        l2.reg*weight[[n.layer + 1]]
      bias[[n.layer + 1]] <- bias[[n.layer + 1]] - db[[n.layer + 1]]
      for(j in n.layer:1) {

        d_h[[j]] <- d_a[[j + 1]] %*% t(weight[[j + 1]])
        d_a[[j]] <- d_h[[j]] * activate_(a[[j]])

        if(j > 1) {
          d_w[[j]] <- t(h[[j - 1]]) %*% d_a[[j]]
        } else {
          d_w[[j]] <- t(xi) %*% d_a[[j]]
        }
        # weight[[j]] <- weight[[j]] - learning.rate * d_w[[j]] -
        #   l1.reg*((weight[[j]] > 0) - (weight[[j]] < 0)) -
        #   l2.reg*weight[[j]]
        # bias[[j]] <- bias[[j]] - learning.rate * (t(one_sample_size[[i]]) %*% d_a[[j]])
        bias.grad <- (t(one_sample_size[[i]]) %*% d_a[[j]])
        if(learning.rate.adaptive == "momentum") {

          last.dw[[j]] <- last.dw[[j]] * rho + d_w[[j]] * learning.rate
          last.db[[j]] <- last.db[[j]] * rho + bias.grad * learning.rate
          dw[[j]] <- last.dw[[j]]
          db[[j]] <- last.db[[j]]
        } else if (learning.rate.adaptive == "adagrad") {

          weight.ss[[j]] <- weight.ss[[j]] + d_w[[j]]**2
          bias.ss[[j]]   <- bias.ss[[j]]   + bias.grad**2
          dw[[j]] <- d_w[[j]]/sqrt(weight.ss[[j]] + epsilon) * learning.rate
          db[[j]] <- bias.grad/sqrt(bias.ss[[j]] + epsilon) * learning.rate
        } else if (learning.rate.adaptive == "adadelta") {

          weight.egs[[j]] <- weight.egs[[j]] * rho + (1-rho) * d_w[[j]]**2
          bias.egs[[j]]   <-   bias.egs[[j]] * rho + (1-rho) * bias.grad**2
          dw[[j]] <- sqrt(weight.es[[j]] + epsilon) / sqrt(weight.egs[[j]] + epsilon) * d_w[[j]]
          db[[j]] <- sqrt(  bias.es[[j]] + epsilon) / sqrt(  bias.egs[[j]] + epsilon) * bias.grad
          weight.es[[j]] <- weight.es[[j]] * rho + (1-rho) * dw[[j]]**2
          bias.es[[j]]   <-   bias.es[[j]] * rho + (1-rho) * db[[j]]**2
        } else if (learning.rate.adaptive == "adam") {

          # mt.ind <- mt.ind + 1
          mt.w[[j]] <- mt.w[[j]] * beta1 + (1-beta1) * d_w[[j]]
          mt.b[[j]] <- mt.b[[j]] * beta1 + (1-beta1) * bias.grad
          vt.w[[j]] <- vt.w[[j]] * beta2 + (1-beta2) * d_w[[j]]**2
          vt.b[[j]] <- vt.b[[j]] * beta2 + (1-beta2) * bias.grad**2
          dw[[j]] <- learning.rate * mt.w[[j]] / (1-beta1**mt.ind) / (sqrt(vt.w[[j]] / (1-beta2**mt.ind)) + epsilon)
          db[[j]] <- learning.rate * mt.b[[j]] / (1-beta1**mt.ind) / (sqrt(vt.b[[j]] / (1-beta2**mt.ind)) + epsilon)
        } else {

          dw[[j]] <- d_w[[j]] * learning.rate
          db[[j]] <- bias.grad * learning.rate
        }
        # browser()
        weight[[j]] <- weight[[j]] - dw[[j]] - l1.reg*((weight[[j]] > 0) - (weight[[j]] < 0)) - l2.reg*weight[[j]]
        bias[[j]]   <- bias[[j]]   - db[[j]]
      }
    }

    if(valid) {

      for(j in 1:n.layer) {

        if(j == 1) {
          pred <- activate(x.valid %*% weight[[j]] + valid_sample_size %*% bias[[j]])
        } else {
          pred <- activate(pred %*% weight[[j]] + valid_sample_size %*% bias[[j]])
        }
      }
      pred <- (pred %*% weight[[n.layer + 1]] + valid_sample_size %*% bias[[n.layer + 1]])[, 1]
      # if(model.type == "classification") {
      if(loss.f == "logit") {
        pred <- 1/(exp(-pred) + 1)
        loss[k] <- -sum(w.valid * (y.valid * log(pred) + (1-y.valid) * log(1-pred))) / sum(w.valid)
      } else if(loss.f == "mse") {
        loss[k] <- sum(w.valid * (y.valid - pred)**2) / sum(w.valid)
      } else if(loss.f == "rmsle") {
        pred <- relu(pred)
        loss[k] <- sum(w.valid * (log(y.valid+1) - log(pred+1))**2) / sum(w.valid)
      }

      if(is.na(loss[k]) | is.null(loss[k]) | is.nan(loss[k]) | is.infinite(loss[k])) {

        loss <- loss[-k]
        break
      } else {

        if(loss[k] < best.loss) {

          best.loss <- loss[k]
          best.weight <- weight
          best.bias <- bias
        }

        if(k > early.stop.det + 1) {
          if(prod(loss[k:(k - early.stop.det + 1)] > loss[(k-1):(k - early.stop.det)]) > 0) {
            break
          }
        }
      }
    }
  }

  if(early.stop) {

    best_weight_ <- best.weight
    best_bias_ <- best.bias
  } else {

    best_weight_ <- weight
    best_bias_ <- bias
  }

  return(list(best_weight_, best_bias_, loss, length(loss)-1))
}








