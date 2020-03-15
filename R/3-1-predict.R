###########################################################
### Predict using dnnet objects

#' @name predict
#' @rdname predict
#'
#' @title Predict function for the package
#'
#' @description The prediction function for \code{dnnet} or \code{dnnetEnsemble} object.
#'
#' @param object A \code{dnnet} or \code{dnnetEnsemble} object.
#' @param newData A matrix with the same number of columns in the input data.
#' @param type Consistent with model.type in the \code{object}.
#' @param ... Other parameters.
#'
#' @return A numeric vector for regression or a matrix of probabilities for each class for classification.
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' \code{\link{dnnetEnsemble-class}}\cr
#'
NULL

#' @rdname predict
#' @export
setMethod("predict",
          "dnnet",
          function(object, newData, type, ...) {

            n.layer <- length(object@bias) - 1
            activate <- get(object@model.spec$activate)
            one_sample_size <- rep(1, dim(newData)[1])
            newData <- (newData - outer(rep(1, dim(newData)[1]), object@norm$x.center)) /
              outer(rep(1, dim(newData)[1]), object@norm$x.scale)

            if(object@model.spec$pathway) {

              activate.p <- get(object@model.spec$pathway.active)
              x_pathway <- list()
              newData0 <- matrix(NA, dim(newData)[1], length(object@model.spec$pathway.list))
              for(i in 1:length(object@model.spec$pathway.list)) {
                x_pathway[[i]] <- newData[, object@model.spec$pathway.list[[i]]]
                newData0[, i] <- activate.p(x_pathway[[i]] %*% object@model.spec$weight.pathway[[i]] +
                                              object@model.spec$bias.pathway[i])
              }
              newData <- cbind(newData0, newData[, -unlist(object@model.spec$pathway.list)])
            }

            for(j in 1:n.layer) {

              if(j == 1) {
                pred <- activate(newData %*% object@weight[[j]] + one_sample_size %*% object@bias[[j]])
              } else {
                pred <- activate(pred %*% object@weight[[j]] + one_sample_size %*% object@bias[[j]])
              }
            }

            if(object@model.type == "binary-classification") {
              pred <- (pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]])[, 1]
              pred <- 1/(exp(-pred) + 1)
              return(matrix(cbind(pred, 1-pred), dim(newData)[1], length(object@label),
                            dimnames = list(NULL, object@label)))
            }

            if(object@model.type == "multi-classification") {
              pred <- pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]]
              pred <- exp(pred)/rowSums(exp(pred))
              return(matrix(pred, dim(newData)[1], length(object@label),
                            dimnames = list(NULL, object@label)))
            }

            if(object@model.type == "ordinal-multi-classification") {
              pred <- pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]]
              pred <- 1/(exp(-pred) + 1)
              first_col <- 1 - pred[, 1]
              for(d in 1:(dim(pred)[2]-1)) pred[, d] <- pred[, d] - pred[, d+1]
              pred <- cbind(first_col, pred)
              return(matrix(pred, dim(newData)[1], length(object@label),
                            dimnames = list(NULL, object@label)))
            }

            if(object@model.type == "multi-regression") {
              pred <- pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]]
              sample.size <- dim(pred)[1]
              return(pred*(rep(1, sample.size) %*% t(object@norm$y.scale)) +
                       rep(1, sample.size) %*% t(object@norm$y.center))
            }

            if(object@model.type %in% c("poisson", "negbin")) {
              pred <- (pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]])[, 1]
              return(exp(pred*object@norm$y.scale + object@norm$y.center))
            }

            if(object@model.type %in% c("zip", "zinb")) {
              pred <- (pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]])
              return(cbind(1/(1+exp(-(pred[, 1]*object@norm$y.scale + object@norm$y.center))),
                           exp(pred[, 2]*object@norm$y.scale + object@norm$y.center)))
            }

            pred <- (pred %*% object@weight[[n.layer + 1]] + one_sample_size %*% object@bias[[n.layer + 1]])[, 1]
            return(pred*object@norm$y.scale + object@norm$y.center)
          })

###########################################################
### Predict using dnnetEnsemble objects

#' @rdname predict
#' @export
setMethod("predict",
          "dnnetEnsemble",
          function(object, newData, type, ...) {

            if(object@model.type %in% c("multi-classification", "ordinal-multi-classification",
                                        "multi-regression")) {
              count <- 0
              for(i in 1:length(object@keep)) {

                if(object@keep[i]) {

                  count <- count + 1
                  pred <- predict(object@model.list[[i]], newData)
                  if(count == 1)
                    pred.all <- array(NA, dim = c(sum(object@keep), dim(pred)))
                  pred.all[count, , ] <- pred
                }
              }

              pred.avg <- apply(pred.all, 2:3, mean)
              return(pred.avg)
            } else if(object@model.type %in% c("zip", "zinb")) {

              count <- 0
              for(i in 1:length(object@keep)) {

                if(object@keep[i]) {

                  count <- count + 1
                  pred <- predict(object@model.list[[i]], newData)
                  if(count == 1)
                    pred.all <- array(NA, dim = c(sum(object@keep), dim(pred)))
                  pred.all[count, , ] <- pred
                }
              }

              pred.avg <- cbind(apply(pred.all[, , 1], 2, function(x) 1/(1+exp(-mean(log(x/(1-x)))))),
                                apply(pred.all[, , 2], 2, function(x) exp(mean(log(x)))))
              return(pred.avg)
            } else {

              pred.all <- c()
              for(i in 1:length(object@keep)) {

                if(object@keep[i]) {

                  pred <- predict(object@model.list[[i]], newData)
                  if(object@model.type == "binary-classification")
                    pred <- pred[, object@model.list[[1]]@label[1]]
                  pred.all <- cbind(pred.all, pred)
                }
              }

              pred.avg <- apply(pred.all, 1, mean)
              if(object@model.type == "binary-classification")
                return(matrix(c(pred.avg, 1 - pred.avg), dim(newData)[1], 2,
                              dimnames = list(NULL, object@model.list[[1]]@label)))
              return(pred.avg)
            }
          })
