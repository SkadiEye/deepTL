###########################################################
### Model passed to PermFIT (Internal)

#' Model passed to PermFIT (Internal)
#'
#' Model passed to PermFIT (Internal)
#'
#' @param method Name of the model.
#' @param model.type Type of model.
#' @param object A dnnetInput or dnnetSurvInput object.
#' @param ... Orger parameters passed to the model.
#'
#' @return Returns a specific model.
#'
#' @importFrom randomForestSRC rfsrc
#' @importFrom randomForest randomForest
#' @importFrom glmnet cv.glmnet
#' @importFrom glmnet glmnet
#' @importFrom stats lm
#' @importFrom stats glm
#' @importFrom survival coxph
#' @importFrom survival Surv
#' @importFrom e1071 svm
#' @importFrom e1071 tune.svm
#'
#' @export
mod_permfit <- function(method, model.type, object, ...) {

  if(method == "ensemble_dnnet") {

    mod <- do.call(ensemble_dnnet, appendArg(list(...), "object", object, TRUE))
  } else if (method == "random_forest") {

    if(model.type == "survival") {
      mod <- do.call(randomForestSRC::rfsrc,
                     appendArg(appendArg(list(...), "formula", Surv(y, e) ~ ., TRUE),
                               "data", data.frame(x = object@x, y = object@y, e = object@e), TRUE))
    } else {
      mod <- do.call(randomForest::randomForest,
                     appendArg(appendArg(list(...), "x", object@x, TRUE), "y", object@y, TRUE))
    }
  } else if (method == "lasso") {

    lasso_family <- ifelse(model.type == "regression", "gaussian",
                           ifelse(model.type == "binary-classification", "binomial", "cox"))
    cv_lasso_mod <- glmnet::cv.glmnet(object@x, object@y, family = lasso_family)
    mod <- glmnet::glmnet(object@x, object@y, family = lasso_family,
                          lambda = cv_lasso_mod$lambda[which.min(cv_lasso_mod$cvm)])
  } else if (method == "linear") {

    if(model.type == "regression") {
      mod <- stats::lm(y ~ ., data.frame(x = object@x, y = object@y))
    } else if(model.type == "binary-classification") {
      mod <- stats::glm(y ~ ., family = "binomial", data = data.frame(x = object@x, y = object@y))
    } else {
      mod <- survival::coxph(survival::Surv(y, e) ~ ., data = data.frame(x = object@x, y = object@y, e = object@e))
    }
  } else if (method == "svm") {

    if(model.type == "regression") {
      mod <- e1071::tune.svm(object@x, object@y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                             tunecontrol = e1071::tune.control(cross = 5))
      mod <- mod$best.model
    } else if(model.type == "binary-classification") {
      mod <- e1071::tune.svm(object@x, object@y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                             tunecontrol = e1071::tune.control(cross = 5))
      mod <- svm(object@x, object@y, gamma = mod$best.parameters$gamma, cost = mod$best.parameters$cost, probability = TRUE)
    } else {
      return("Not Applicable")
    }
  } else if (method == "dnnet") {

    spli_obj <- splitDnnet(object, 0.8)
    mod <- do.call(dnnet, appendArg(appendArg(list(...), "train", spli_obj$train, TRUE),
                                    "validate", spli_obj$valid, TRUE))
  } else {

    return("Not Applicable")
  }
  return(mod)
}

###########################################################
### Model prediction passed to PermFIT (Internal)

#' Model prediction passed to PermFIT (Internal)
#'
#' Model prediction passed to PermFIT (Internal)
#'
#' @param mod Model for prediction.
#' @param object A dnnetInput or dnnetSurvInput object.
#' @param method Name of the model.
#' @param model.type Type of the model.
#'
#' @return Returns predictions.
#'
#' @export
predict_mod_permfit <- function(mod, object, method, model.type) {

  if(model.type == "regression") {

    if(!method %in% c("linear", "lasso")) {
      return(predict(mod, object@x))
    } else if(method == "linear") {
      return(predict(mod, object@x)[, "s0"])
    } else {
      return(predict(mod, object@x)[, "s0"])
    }
  } else if(model.type == "binary-classification") {

    if(method %in% c("dnnet")) {
      return(predict(mod, object@x)[, mod@label[1]])
    } else if(method == "ensemble_dnnet") {
      return(predict(mod, object@x)[, mod@model.list[[1]]@label[1]])
    } else if(method == "random_forest") {
      return(predict(mod, object@x, type = "prob")[, 1])
    } else if (method == "lasso") {
      return(1 - predict(mod, object@x, type = "response")[, "s0"])
    } else if (method == "linear") {
      return(1 - predict(mod, data.frame(x = object@x, y = object@y), type = "response"))
    } else if (method == "svm") {
      return(attr(predict(mod, object@x, decision.values = TRUE, probability = TRUE),
                  "probabilities")[, levels(object@y)[1]])
    }
  } else if(model.type == "survival") {

    if(method %in% c("ensemble_dnnet", "dnnet")) {
      return(predict(mod, object@x))
    } else if(method == "random_forest") {
      return(log(predict(mod, data.frame(x = object@x))$predicted))
    } else if (method == "lasso") {
      return(predict(mod, object@x)[, "s0"])
    } else if (method == "linear") {
      return(predict(mod, data.frame(x = object@x), type = "lp"))
    } else if (method == "svm") {
      return("Not Applicable")
    }
  } else {
    return("Not Applicable")
  }
}

###########################################################
### Log-likelihood for Cox-Model (Internal)

#' Log-likelihood for Cox-Model (Internal)
#'
#' Log-likelihood for Cox-Model (Internal)
#'
#' @param h Log hazard.
#' @param y Event time.
#' @param e Event status.
#'
#' @return Returns log-likelihood.
#'
#' @export
cox_logl <- function(h, y, e) {
  order_y <- order(y)
  n <- length(y)
  map_ <- ((rep(1, n) %*% t(1:n)) >= (1:n %*% t(rep(1, n))))*1
  # sum((h[order_y] - log(colSums(exp(rep(1, n) %*% t(h[order_y]))*map_)))*e[order_y])
  (h[order_y] - log(colSums(exp(rep(1, n) %*% t(h[order_y]))*map_)))*e[order_y]
}

###########################################################
### Log-likelihood Difference (Internal)

#' Log-likelihood Difference (Internal)
#'
#' Log-likelihood Difference (Internal)
#'
#' @param model.type Type of the model.
#' @param y_hat Y hat.
#' @param y_hat0 Another Y hat.
#' @param object Data object.
#'
#' @return Returns log-likelihood difference.
#'
#' @export
log_lik_diff <- function(model.type, y_hat, y_hat0, object, y_max = 1-10**-10, y_min = 10**-10) {

  if(model.type == "regression") {
    return((object@y - y_hat)**2 - (object@y - y_hat0)**2)
  } else if(model.type == "binary-classification") {
    y_hat <- ifelse(y_hat < y_min, y_min, ifelse(y_hat > y_max, y_max, y_hat))
    y_hat0 <- ifelse(y_hat0 < y_min, y_min, ifelse(y_hat0 > y_max, y_max, y_hat0))
    return(-(object@y == levels(object@y)[1])*log(y_hat) - (object@y != levels(object@y)[1])*log(1-y_hat) +
             (object@y == levels(object@y)[1])*log(y_hat0) + (object@y != levels(object@y)[1])*log(1-y_hat0))
  } else if(model.type == "survival") {
    return(cox_logl(y_hat, object@y, object@e) -
             cox_logl(y_hat0, object@y, object@e))
  } else {
    return("Not Applicable")
  }
}

###########################################################
### PermFIT

#' PermFIT: A permutation-based feature importance test.
#'
#' @param train An dnnetInput or dnnetSurvInput object.
#' @param validate A validation dataset is required when k_fold = 0.
#' @param k_fold K-fold cross-fitting. If not, set k_fold to zero.
#' @param n_perm Number of permutations repeated.
#' @param pathway_list A list of pathways to be jointly tested.
#' @param method Models, including \code{ensemble_dnnet} for ensemble deep
#'   neural networks, \code{random_forest} for random forests or random
#'   survival forests, \code{lasso} for linear/logistic/cox lasso, \
#'   \code{linear} for linear/logistic/coxph regression, \code{svm}
#'   for svms with Gaussian kernels, and \code{dnnet} with single deep
#'   neural network.
#' @param shuffle If shuffle is null, the data will be shuffled for
#'   cross-fitting; if random shuffle is not desired, please provide
#'   a bector of numbers for cross-fitting indices
#' @param ... Additional parameters passed to each method.
#'
#' @return Returns a PrmFIT object.
#'
#' @importFrom stats sd
#' @importFrom stats pnorm
#' @importFrom stats var
#'
#' @export
permfit <- function(train, validate = NULL, k_fold = 5,
                    n_perm = 100, pathway_list = list(),
                    method = c("ensemble_dnnet", "random_forest",
                               "lasso", "linear", "svm", "dnnet")[1],
                    shuffle = NULL,
                    ...) {
  n_pathway <- length(pathway_list)
  n <- dim(train@x)[1]
  p <- dim(train@x)[2]
  if(class(train) == "dnnetSurvInput") {
    model.type <- "survival"
  } else if(class(train) == "dnnetInput") {
    if(is.factor(train@y)) {
      model.type <- "binary-classification"
    } else {
      model.type <- "regression"
    }
  } else {
    stop("'train' has to be either a dnnetInput or dnnetSurvInput object.")
  }

  if(k_fold == 0) {

    if(is.null(validate))
      stop("A validation set is required when k = 0. ")
    n_valid <- dim(validate@x)[1]

    mod <- mod_permfit(method, model.type, train, ...)
    f_hat_x <- predict_mod_permfit(mod, validate, method, model.type)
    valid_ind <- list(1:length(validate@y))
    y_pred <- f_hat_x

    if(n_pathway >= 1) {
      p_score <- array(NA, dim = c(n_perm, n_valid, n_pathway))
      for(i in 1:n_pathway) {
        for(l in 1:n_perm) {

          x_i <- validate@x
          x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(n_valid), ]
          pred_i <- predict_mod_permfit(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
          p_score[l, , i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
        }
      }
    }

    p_score2 <- array(NA, dim = c(n_perm, n_valid, p))
    for(i in 1:p) {
      for(l in 1:n_perm) {

        x_i <- validate@x
        x_i[, i] <- x_i[, i][sample(n_valid)]
        pred_i <- predict_mod_permfit(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
        p_score2[l, , i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
      }
    }
  } else {

    valid_ind <- list()
    if(is.null(shuffle)) shuffle <- sample(n)
    n_valid <- n
    y_pred <- numeric(length(train@y))
    if(n_pathway >= 1)
      p_score <- array(NA, dim = c(n_perm, n_valid, n_pathway))
    p_score2 <- array(NA, dim = c(n_perm, n_valid, p))
    valid_error <- numeric(k_fold)
    for(k in 1:k_fold) {

      train_spl <- splitDnnet(train, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)])
      valid_ind[[k]] <- shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)]

      mod <- mod_permfit(method, model.type, train_spl$valid, ...)
      f_hat_x <- predict_mod_permfit(mod, train_spl$train, method, model.type)
      valid_error[k] <- sum(log_lik_diff(model.type, f_hat_x, f_hat_x, train_spl$train))
      y_pred[valid_ind[[k]]] <- f_hat_x
      if(k == 1) {

        final_model <- mod
      } else if(method == "ensemble_dnnet") {

        final_model@model.list <- c(final_model@model.list, mod@model.list)
        final_model@loss <- c(final_model@loss, mod@loss)
        final_model@keep <- c(final_model@keep, mod@keep)
      }

      if(n_pathway >= 1) {
        for(i in 1:n_pathway) {
          for(l in 1:n_perm) {

            x_i <- train_spl$train@x
            x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(dim(x_i)[1]), ]
            pred_i <- predict_mod_permfit(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
            p_score[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], i] <- log_lik_diff(model.type, pred_i, f_hat_x, train_spl$train)
          }
        }
      }

      for(i in 1:p) {
        for(l in 1:n_perm) {

          x_i <- train_spl$train@x
          x_i[, i] <- x_i[, i][sample(dim(x_i)[1])]
          pred_i <- predict_mod_permfit(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
          p_score2[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], i] <- log_lik_diff(model.type, pred_i, f_hat_x, train_spl$train)
        }
      }
    }
    mod <- final_model
    valid_error <- sum(valid_error)/n_valid
  }

  if(is.null(colnames(train@x))) {
    imp <- data.frame(var_name = paste0("V", 1:p))
  } else  {
    imp <- data.frame(var_name = colnames(train@x))
  }
  imp$importance <- apply(apply(p_score2, 2:3, mean), 2, mean, na.rm = TRUE)
  imp$importance_sd <- sqrt(apply(apply(p_score2, 2:3, mean), 2, stats::var, na.rm = TRUE)/n_valid)
  imp$importance_pval <- 1 - stats::pnorm(imp$importance/imp$importance_sd)
  if(n_perm > 1) {
    imp$importance_sd_x <- apply(apply(p_score2, c(1, 3), mean), 2, stats::sd, na.rm = TRUE)
    imp$importance_pval_x <- 1 - stats::pnorm(imp$importance/imp$importance_sd_x)
  }

  imp_block <- data.frame()
  if(n_pathway >= 1) {

    if(is.null(names(pathway_list))) {
      imp_block <- data.frame(block = paste0("P", 1:n_pathway))
    } else {
      imp_block <- data.frame(block = names(pathway_list))
    }
    imp_block$importance <- apply(apply(p_score, 2:3, mean), 2, mean, na.rm = TRUE)
    imp_block$importance_sd <- sqrt(apply(apply(p_score, 2:3, mean), 2, stats::var, na.rm = TRUE)/n_valid)
    imp_block$importance_pval <- 1 - stats::pnorm(imp_block$importance/imp_block$importance_sd)
    if(n_perm > 1) {
      imp_block$importance_sd_x <- apply(apply(p_score, c(1, 3), mean), 2, stats::sd, na.rm = TRUE)
      imp_block$importance_pval_x <- 1 - stats::pnorm(imp_block$importance/imp_block$importance_sd_x)
    }
  }

  return(new("PermFIT", model = mod, importance = imp, block_importance = imp_block,
             validation_index = valid_ind, y_hat = y_pred))
}

