###########################################################
### Double deep treatment learning

#' The algorithm for double deep treatment learning
#'
#' The algorithm for double deep treatment learning in comparative effectiveness analysis.
#'
#' @param object A \code{dnnetInput} or a \code{dnnetSurvInput} object, the training set.
#' @param en_dnn_ctrl1 A list of parameters to be passed to \code{ensemble_dnnet} for model 1.
#' @param en_dnn_ctrl2 A list of parameters to be passed to \code{ensemble_dnnet} for model 2.
#' @param methods Methods included in this analysis.
#' @param ... Parameters passed to both en_dnn_ctrl1 and en_dnn_ctrl2.
#'
#' @return Returns a \code{list} of results.
#'
#' @importFrom stats lm
#' @importFrom stats coefficients
#' @importFrom stats vcov
#'
#' @export
double_deepTL <- function(object,
                          en_dnn_ctrl1 = NULL,
                          en_dnn_ctrl2 = NULL,
                          methods = c("revised-semi", "semi", "cov-adj",
                                      "ipw1", "ipw2", "ipw3", "double-robust"), ...) {

  if(is.null(en_dnn_ctrl1))
    en_dnn_ctrl1 <- list(n.ensemble = 100, verbose = FALSE,
                         esCtrl = list(n.hidden = 10:5*2, n.batch = 100, n.epoch = 100, norm.x = TRUE,
                                       norm.y = TRUE, activate = "relu", accel = "rcpp", l1.reg = 10**-4,
                                       plot = FALSE, learning.rate.adaptive = "adam", early.stop.det = 100))
  if(is.null(en_dnn_ctrl2))
    en_dnn_ctrl2 <- list(n.ensemble = 100, verbose = FALSE,
                         esCtrl = list(n.hidden = 10:5*2, n.batch = 100, n.epoch = 250, norm.x = TRUE,
                                       norm.y = TRUE, activate = "relu", accel = "rcpp", l1.reg = 10**-4,
                                       plot = FALSE, learning.rate.adaptive = "adam", early.stop.det = 100))

  if(!is.factor(object@z))
    object@z <- factor(ifelse(object@z == 1, "A", "B"), levels = c("A", "B"))
  z_obj <- importDnnet(x = object@x, y = object@z)

  result <- data.frame()

  cat("Fitting E(Z|X) ... \n")
  z_mod <- do.call("ensemble_dnnet", appendArg(en_dnn_ctrl1, "object", z_obj, TRUE))
  z_pred <- predict(z_mod, object@x)[, levels(object@z)[1]]
  z_nmrc <- (object@z == levels(object@z)[1]) * 1

  if("revised-semi" %in% methods || "cov-adj" %in% methods) {

    lm_mod <- stats::lm(object@y ~ z_nmrc + z_pred)
    beta1 <- stats::coefficients(lm_mod)[2]
    if("cov-adj" %in% methods)
      result <- rbind(result, data.frame(method = "cov-adj-dnn-en", beta = beta1, var = stats::vcov(lm_mod)[2, 2]))
    if("revised-semi" %in% methods) {

      cat("Fitting E(Y - b1*Z|X) ... \n")
      y_star_obj <- importDnnet(x = object@x, y = object@y - beta1 * z_nmrc)
      y_star_mod <- do.call("ensemble_dnnet", appendArg(en_dnn_ctrl2, "object", y_star_obj, TRUE))
      y_star_pred <- predict(y_star_mod, object@x)
      beta_est <- sum((object@y - beta1 * z_nmrc - y_star_pred) * (z_nmrc - z_pred)) / sum((z_nmrc - z_pred)**2) + beta1
      beta_var <- mean(((object@y - beta1 * z_nmrc - y_star_pred) - (beta_est - beta1) * (z_nmrc - z_pred))**2) / sum((z_nmrc - z_pred)**2)
      result <- rbind(result, data.frame(method = "revised-semi-dnn-en", beta = beta_est, var = beta_var))
    }
  }

  if("semi" %in% methods) {

    cat("Fitting E(Y|X) ... \n")
    y_obj <- importDnnet(x = object@x, y = object@y)
    y_mod <- do.call("ensemble_dnnet", appendArg(en_dnn_ctrl2, "object", y_obj, TRUE))
    y_pred <- predict(y_mod, object@x)
    beta_est <- sum((object@y - y_pred) * (z_nmrc - z_pred)) / sum((z_nmrc - z_pred)**2)
    beta_var <- mean(((object@y - y_pred) - beta_est * (z_nmrc - z_pred))**2) / sum((z_nmrc - z_pred)**2)
    result <- rbind(result, data.frame(method = "semi-dnn-en", beta = beta_est, var = beta_var))
  }

  return(result)
}
