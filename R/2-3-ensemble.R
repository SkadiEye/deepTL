###########################################################
### Negative log-likelihood (Internal)

#' Negative log-likelihood (Internal)
#'
#' @param model.type Type of the model.v
#' @param y_hat Y hat.x
#' @param alpha Other distributional parameter.
#'
#' @return Returns log-likelihood
#'
#' @export
neg_log_lik_ <- function(model.type, y, y_hat, alpha = NULL) {
  if(model.type == "regression") return((y - y_hat)**2)
  if(model.type == "binary-classification") return(-(y*log(y_hat) + (1-y)*log(1-y_hat)))
  if(model.type == "poisson") return(-(y*log(y_hat) - y_hat))
  if(model.type == "poisson-nonzero") return(-(y*log(y_hat) - y_hat - log(1 - exp(-y_hat))))
  if(model.type == "zip") {
    z <- (y > 0)*1
    pi_est <- y_hat[, 1]
    lambda_est <- y_hat[, 2]
    return(-(1-z)*log(1 - pi_est + pi_est*exp(-lambda_est)) -
             z*(pi_est + y*log(lambda_est) - lambda_est))
  }
  if(model.type == "negbin")
    return(-(y*log(y_hat) - (y + 1/alpha)*log(1 + alpha*y_hat)))
  if(model.type == "poisson-nonzero")
    return(-(y*log(y_hat) - (y + 1/alpha)*log(1 + alpha*y_hat) - log(1 - (1 + alpha*y_hat)**(-1/alpha))))
  if(model.type == "zinb") {
    z <- (y > 0)*1
    pi_est <- y_hat[, 1]
    mu_est <- y_hat[, 2]
    return(-(1-z)*log(1 - pi_est + pi_est*(1 + alpha*mu_est)**(-1/alpha)) -
             z*(pi_est + y*log(mu_est) - (y + 1/alpha)*log(1 + alpha*mu_est) - lgamma(1/alpha)))
  }
  if(model.type == "gamma") return(y/y_hat + log(y_hat))
  return("Not Applicable")
}

###########################################################
### DNN ensemble

#' An ensemble model of DNNs
#'
#' Fit a an ensemble model of DNNs
#'
#' @param object A \code{dnnetInput} or a \code{dnnetSurvInput} object, the training set.
#' @param n.ensemble The number of DNNs to be fit
#' @param esCtrl A list of parameters to be passed to \code{dnnet}.
#' @param random.nbatch An indicator whether n.batch is randomly selected.
#' @param random.nbatch.limit Minimum and maximum for randomly chosen n.batch.
#' @param unbalance.trt Treatment for unbalanced labels.
#' @param bootstrap Indicator for whether a bootstrap sampling is used.
#' @param prop.train If bootstrap == FALSE, a training/validation cut for the data will be used (0, 1).
#' @param prop.keep The proportion of DNNs to be kept in the ensemble.
#' @param best.opti Whether to run the algorithm to keep the optimal subset of DNNs.
#' @param min.keep Minimal number of DNNs to be kept.
#' @param verbose Whether the progress to be printed.
#'
#' @return Returns a \code{dnnetEnsemble} object.
#'
#' @seealso
#' \code{\link{dnnetEnsemble-class}}\cr
#' \code{\link{dnnetInput-class}}\cr
#'
#' @export
ensemble_dnnet <- function(object,
                           n.ensemble = 100,
                           esCtrl,
                           random.nbatch = FALSE,
                           random.nbatch.limit = NULL,
                           unbalance.trt = c("None", "Over", "Under", "Balance")[1],
                           bootstrap = TRUE,
                           prop.train = 1,
                           prop.keep = 1,
                           best.opti = TRUE,
                           min.keep = 10,
                           verbose = TRUE) {

  OOB.error <- list()
  model.list <- list()
  # model.type <- ifelse(class(object@y) == "factor", "Binary", "Continuous")
  pred.table <- matrix(NA, n.ensemble, length(object@y))
  loss <- numeric(length(n.ensemble))
  keep <- rep(TRUE, n.ensemble)
  min.keep <- max(min(min.keep, n.ensemble), 1)

  if(verbose)
    pb <- utils::txtProgressBar(min = 0, max = n.ensemble, style = 3)
  for(i in 1:n.ensemble) {

    if(class(object) != "RegObj" && unbalance.trt != "None") {

      y.table <- table(object@y)
      class.size <- ifelse(unbalance.trt == "Over", max(y.table),
                           ifelse(unbalance.trt == "Under", min(y.table),
                                  ifelse(unbalance.trt == "Balance", mean(y.table), stop("Invalid input for Unbalance Treatment. "))))
      train.ind2 <- c()
      for(k in 1:length(unique(object@y)))
        train.ind2 <- c(train.ind2, sample(which(object@y == names(y.table)[k]), class.size, replace = TRUE))

      train.boot <- splitDnnet(object, train.ind2)
    } else if (bootstrap) {

      train.boot <- splitDnnet(object, "bootstrap")
    } else {

      train.boot <- splitDnnet(object, prop.train)
    }

    trainObj <- train.boot$train
    validObj <- train.boot$valid
    trainObj.ind <- train.boot$split
    args <- esCtrl # removeArg(esCtrl, "machine")
    if(random.nbatch)
      args <- appendArg(args, "n.batch", ceiling(exp(runif(1, log(random.nbatch.limit[1]), log(random.nbatch.limit[2])))), TRUE)
    args <- appendArg(args, "validate", validObj, 1)
    args <- appendArg(args, "train", trainObj, 1)

    model <- do.call(deepTL::dnnet, args)
    model.list[[i]] <- model
    if(i == 1) model.type <- model@model.type
    if(i == 1) model.spec <- model@model.spec
    if((model.type %in% c("negbin", "negbin-nonzero", "zinb")) && (i > 1))
      model.spec$negbin_alpha <- c(model.spec$negbin_alpha, model@model.spec$negbin_alpha)
    if((model.type == "gamma") && (i > 1))
      model.spec$gamma_alpha <- c(model.spec$gamma_alpha, model@model.spec$gamma_alpha)

    pred <- predict(model, object@x)
    if(model.type == "regression") {

      pred.table[i, ] <- pred
      loss[i] <- sum(((validObj@y - mean(validObj@y))**2 - (validObj@y - pred[-trainObj.ind])**2) *
                       validObj@w) / sum(validObj@w)

    } else if(model.type == "binary-classification") {

      pred.table[i, ] <- pred[, levels(validObj@y)[1]]
      loss[i] <- sum((-log(mean(validObj@y == levels(validObj@y)[1])) * (validObj@y == levels(validObj@y)[1]) -
                        log(1-mean(validObj@y == levels(validObj@y)[1])) * (1-(validObj@y == levels(validObj@y)[1])) +
                        log(pred[-trainObj.ind, levels(validObj@y)[1]]) * (validObj@y == levels(validObj@y)[1]) +
                        log(1-pred[-trainObj.ind, levels(validObj@y)[1]]) * (1-(validObj@y == levels(validObj@y)[1]))) *
                       validObj@w) / sum(validObj@w)
    } else if(model.type == "multi-classification") {

      if(i == 1) {

        pred.table <- array(NA, dim = c(n.ensemble, dim(pred)))
        mat_y <- matrix(NA, dim(pred)[1], dim(pred)[2])
        for(d in 1:dim(pred)[2]) mat_y[, d] <- (object@y == levels(object@y)[d])*1
      }
      pred.table[i, , ] <- pred
      loss[i] <- -sum((mat_y[-trainObj.ind, ] * log(pred[-trainObj.ind, ]) - mat_y[-trainObj.ind, ] *
                         (rep(1, length(validObj@y)) %*% t(log(colMeans(mat_y[-trainObj.ind, ]))))) *
                        validObj@w) / sum(validObj@w)
    } else if(model.type == "ordinal-multi-classification") {

      for(d in (dim(pred)[2]-1):1) pred[, d] <- pred[, d] + pred[, d+1]
      pred <- pred[, -1]
      if(i == 1) {

        pred.table <- array(NA, dim = c(n.ensemble, dim(pred)))
        mat_y <- matrix(0, dim(pred)[1], dim(pred)[2] + 1)
        for(d in 1:dim(pred)[1]) mat_y[d, 1:match(object@y[d], levels(object@y))] <- 1
        mat_y <- mat_y[, -1]
      }
      pred.table[i, , ] <- pred
      loss[i] <- -sum((mat_y[-trainObj.ind, ] * log(pred[-trainObj.ind, ]) +
                         (1-mat_y[-trainObj.ind, ]) * log(1-pred[-trainObj.ind, ])) *
                        validObj@w) / sum(validObj@w)
    } else if(model.type == "survival") {

      pred.table[i, ] <- pred
      y_order <- order(validObj@y)
      h_y_order <- pred[-trainObj.ind][y_order]
      e_y_order <- validObj@e[y_order]
      curr <- sum(exp(h_y_order))
      loss[i] <- 0
      for(j in 1:length(validObj@y)) {

        if(e_y_order[j] == 1)
          loss[i] <- loss[i] - (h_y_order[j] - log(curr))
        curr <- curr - exp(h_y_order[j])
      }
      if(sum(e_y_order) > 0)
        loss[i] <- loss[i]/sum(e_y_order)
    } else if(model.type %in% c("poisson", "poisson-nonzero", "gamma")) {

      pred.table[i, ] <- pred
      loss[i] <- sum((neg_log_lik_(model.type, validObj@y, rep(mean(validObj@y), length(validObj@y))) -
                        neg_log_lik_(model.type, validObj@y, pred[-trainObj.ind]))*validObj@w) / sum(validObj@w)
    } else if(model.type == "zip") {

      if(i == 1) pred.table <- array(NA, dim = c(n.ensemble, dim(pred)))
      pred.table[i, , ] <- pred
      loss[i] <- sum((neg_log_lik_(model.type, validObj@y,
                                   cbind(rep(mean(validObj@y > 0), length(validObj@y)),
                                         rep(mean(validObj@y)/mean(validObj@y > 0), length(validObj@y)))) -
                        neg_log_lik_(model.type, validObj@y, pred[-trainObj.ind, ]))*validObj@w) / sum(validObj@w)
    } else if(model.type %in% c("negbin", "negbin-nonzero")) {

      pred.table[i, ] <- pred
      if(i == n.ensemble) {
        negbin_alpha <- exp(mean(log(model.spec$negbin_alpha)))
        for(iii in 1:n.ensemble)
          loss[iii] <- sum((neg_log_lik_(model.type, validObj@y, rep(mean(validObj@y), length(validObj@y)), alpha = negbin_alpha) -
                              neg_log_lik_(model.type, validObj@y, pred.table[iii, -trainObj.ind], alpha = negbin_alpha))*validObj@w) / sum(validObj@w)
      }
    } else if(model.type == "zinb") {

      if(i == 1) pred.table <- array(NA, dim = c(n.ensemble, dim(pred)))
      pred.table[i, , ] <- pred
      if(i == n.ensemble) {
        negbin_alpha <- exp(mean(log(model.spec$negbin_alpha)))
        for(iii in 1:n.ensemble)
          loss[iii] <- sum((neg_log_lik_(model.type, validObj@y,
                                         cbind(rep(mean(validObj@y > 0), length(validObj@y)),
                                               rep(mean(validObj@y)/mean(validObj@y > 0), length(validObj@y))), alpha = negbin_alpha) -
                              neg_log_lik_(model.type, validObj@y, pred.table[iii, -trainObj.ind, ], alpha = negbin_alpha))*validObj@w) / sum(validObj@w)
      }
    }

    if(verbose)
      utils::setTxtProgressBar(pb, i)
  }
  if(verbose)
    close(pb)

  if(best.opti && model.type == "regression") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(sum((object@y - colMeans(pred.table[loss >= sort(loss)[i], ]))**2 *
                          object@w) / sum(object@w), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == "binary-classfication") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(-sum((object@y == levels(validObj@y)[1]) * log(colMeans(pred.table[loss >= sort(loss)[i], ])) +
                           (1-(object@y == levels(validObj@y)[1])) * log(1-colMeans(pred.table[loss >= sort(loss)[i], ])) *
                           object@w) / sum(object@w), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == "multi-classfication") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(-sum(mat_y * log(apply(pred.table[loss >= sort(loss)[i], , ], 2:3, mean)) *
                           object@w) / sum(object@w), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == "ordinal-multi-classification") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(-sum((mat_y * log(apply(pred.table[loss >= sort(loss)[i], , ], 2:3, mean)) +
                            (1-mat_y) * log(1-apply(pred.table[loss >= sort(loss)[i], , ], 2:3, mean))) *
                           object@w) / sum(object@w), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == "survival") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse)) {
      pred <- colMeans(pred.table[loss >= sort(loss)[i], ])
      y_order <- order(object@y)
      h_y_order <- pred[y_order]
      e_y_order <- object@e[y_order]
      curr <- sum(exp(h_y_order))
      mse[i] <- 0
      for(j in 1:length(validObj@y)) {

        if(e_y_order[j] == 1)
          mse[i] <- mse[i] - (h_y_order[j] - log(curr))
        curr <- curr - exp(h_y_order[j])
      }
      if(sum(e_y_order) > 0)
        mse[i] <- mse[i]/sum(e_y_order)

    }

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type %in% c("poisson", "poisson-nonzero", "gamma")) {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- sum(neg_log_lik_(model.type, object@y, colMeans(pred.table[loss >= sort(loss)[i], ]))*object@w)/sum(object@w)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == c("zip")) {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- sum(neg_log_lik_(model.type, object@y, apply(pred.table[loss >= sort(loss)[i], , ], 2:3, mean))*object@w)/sum(object@w)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type %in% c("negbin", "negbin-nonzero")) {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- sum(neg_log_lik_(model.type, object@y, colMeans(pred.table[loss >= sort(loss)[i], ]),
                                 alpha = negbin_alpha)*object@w)/sum(object@w)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == c("zinb")) {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- sum(neg_log_lik_(model.type, object@y, apply(pred.table[loss >= sort(loss)[i], , ], 2:3, mean),
                                 alpha = negbin_alpha)*object@w)/sum(object@w)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  }

  if(prop.keep[1] < 1)
    keep <- loss >= sort(loss)[min(max(floor(n.ensemble*(1-prop.keep[1])), 1), n.ensemble - min.keep + 1)]
  if(length(prop.keep) > 1)
    for(i in 2:length(prop.keep))
      keep <- rbind(keep, loss >= sort(loss)[min(max(floor(n.ensemble*(1-prop.keep[i])), 1), n.ensemble - min.keep + 1)])

  return(methods::new("dnnetEnsemble",
                      model.list = model.list,
                      model.type = model.type,
                      model.spec = model.spec,
                      loss = loss,
                      keep = keep))
}
