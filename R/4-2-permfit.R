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
      mod <- lm(y ~ ., data.frame(x = object@x, y = object@y))
    } else if(model.type == "binary-classification") {
      mod <- glm(y ~ ., family = "binomial", data = data.frame(x = object@x, y = object@y))
    } else {
      mod <- coxph(Surv(y, e) ~ ., data = data.frame(x = train@x, y = train@y, e = train@e))
    }
  } else if (method == "svm") {

    mod <- e1071::tune.svm(object@x, object@y, gamma = 10**(-(0:4)), cost = 10**(0:4/2),
                           tunecontrol = e1071::tune.control(cross = 5))
    mod <- mod$best.model
  } else if (method == "dnnet") {

    spli_obj <- splitDnnet(train, 0.8)
    mod <- do.call(dnnet, appendArg(appendArg(list(...), "train", spli_obj$train, TRUE),
                                    "validate", spli_obj$valid, TRUE))
  } else {

    return("Not Applicable")
  }
  return(mod)
}

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

    if(method %in% c("ensemble_dnnet", "dnnet")) {
      return(predict(mod, object@x)[, dnn_mod_bnry@label[1]])
    } else if(method == "random_forest") {
      return(predict(mod, object@x, type = "prob")[, 1])
    } else if (method == "lasso") {
      return(1 - predict(mod, object@x, type = "response")[, "s0"])
    } else if (method == "linear") {
      return(1 - predict(mod, data.frame(x = object@x, y = object@y), type = "response"))
    } else if (method == "svm") {
      return(attr(predict(mod, object@x, decision.values = TRUE, probability = TRUE),
                  "probabilities")[, levels(train@y)[1]])
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

cox_logl <- function(h, y, e) {
  order_y <- order(y)
  n <- length(y)
  map_ <- ((rep(1, n) %*% t(1:n)) >= (1:n %*% t(rep(1, n))))*1
  sum((h[order_y] - log(colSums(exp(rep(1, n) %*% t(h[order_y]))*map_)))*e[order_y])
}

log_lik_diff <- function(model.type, y_hat, y_hat0, object) {

  if(model.type == "regression") {
    return((object@y - y_hat)**2 - (object@y - y_hat0)**2)
  } else if(model.type == "binary-classification") {
    return(-object@y*log(y_hat) - (1-object@y)*log(1-y_hat) +
             object@y*log(y_hat0) + (1-object@y)*log(1-y_hat0))
  } else if(model.type == "survival") {
    return(cox_logl(y_hat, object@y, object@e) -
             cox_logl(y_hat0, object@y, object@e))
  } else {
    return("Not Applicable")
  }
}

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
      p_score <- array(0, dim = c(n_perm, n_valid, n_pathway))
      for(i in 1:n_pathway) {
        for(l in 1:n_perm) {

          x_i <- validate@x
          x_i[, pathway_list[[i]]] <- x_i[, pathway_list[[i]]][sample(n_valid), ]
          pred_i <- predict_mod_permfit(mod, importDnnet(x = x_i, y = validate@y), method, model.type)
          p_score[l, , i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
        }
      }
    }

    p_score2 <- array(0, dim = c(n_perm, n_valid, p))
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
      p_score <- array(0, dim = c(n_perm, n_valid, n_pathway))
    p_score2 <- array(0, dim = c(n_perm, n_valid, p))
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
            p_score[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
          }
        }
      }

      for(i in 1:p) {
        for(l in 1:n_perm) {

          x_i <- train_spl$train@x
          x_i[, i] <- x_i[, i][sample(dim(x_i)[1])]
          pred_i <- predict_mod_permfit(mod, importDnnet(x = x_i, y = train_spl$train@y), method, model.type)
          p_score2[l, shuffle[floor((k-1)*n/k_fold+1):floor(k*n/k_fold)], i] <- log_lik_diff(model.type, pred_i, f_hat_x, validate)
        }
      }
    }
    mod <- final_model
    valid_error <- sum(valid_error)/n_valid
  }

  if(is.null(rownames(train@x))) {
    imp <- data.frame(var_name = paste0("V", 1:p))
  } else  {
    imp <- data.frame(var_name = rownames(train@x))
  }
  imp$importance <- apply(apply(p_score2, 2:3, mean), 2, mean)
  imp$importance_sd <- sqrt(apply(apply(p_score2, 2:3, mean), 2, var)/n_valid)
  imp$importance_pval <- 1 - pnorm(imp$importance/imp$importance_sd)
  if(n_perm > 1) {
    imp$importance_sd_x <- apply(apply(p_score2, c(1, 3), mean), 2, sd)
    imp$importance_pval_x <- 1 - pnorm(imp$importance/imp$importance_sd_x)
  }

  imp_block <- data.frame()
  if(n_pathway >= 1) {

    if(is.null(names(pathway_list))) {
      imp_block <- data.frame(block = paste0("P", 1:n_pathway))
    } else {
      imp_block <- data.frame(block = names(pathway_list))
    }
    imp_block$importance <- apply(apply(p_score, 2:3, mean), 2, mean)
    imp_block$importance_sd <- sqrt(apply(apply(p_score, 2:3, mean), 2, var)/n_valid)
    imp_block$importance_pval <- 1 - pnorm(imp_block$importance/imp_block$importance_sd)
    if(n_perm > 1) {
      imp_block$importance_sd_x <- apply(apply(p_score, c(1, 3), mean), 2, sd)
      imp_block$importance_pval_x <- 1 - pnorm(imp_block$importance/imp_block$importance_sd_x)
    }
  }

  return(new("PermFIT", model = mod, importance = imp, block_importance = imp_block,
             validation_index = valid_ind, y_hat = y_pred))
}

