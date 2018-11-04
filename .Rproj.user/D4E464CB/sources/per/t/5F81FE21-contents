################################################
#### Ensemble Learning to extend single model

#' Ensemble Learning to extend single model
#'
#' Ensemble Learning to extend a \code{reg}, \code{wcls} or a \code{cvGrid} method. Multiple models
#'  will be fitted using randomly generated bootstrap samples. The out-of-bag (OOB) samples will be used as
#'  the validation set if the corresponding method could use the validation to fit the model. The
#'  predictions from all models will be aggregated when predicting for new data. Each model will be
#'  evaluated and assigned a performance score.
#'
#' @param object A \code{RegObj} or \code{wClsObj} containing the training set.
#' @param n.ensemble The number of models in the ensemble.
#' @param cutoff Cutoff used when predicting OOB samples, passed to \code{cutoff.calc} function.
#' @param esCtrl A \code{list} of arguments that specifies a regression or classification method, handled by
#'  \code{\link{esCtrlPanel}}.
#' @param unbalance.trt A character string that specifies if sampleing techniques should be applied to
#'  data with unbalanced numbers for each class.
#' @param prop.keep A \code{numeric} value that specifies the percentage of models to keep with top performance scores.
#' @param best.opti A \code{logical} value, whether the number of model to keep is determined automatically.
#' @param min.keep A \code{numeric} value that specifies the minimum models to keep.
#'
#' @return A \code{\link{ModelEnsembleObj-class}} object.
#'
#' @seealso
#' \code{\link{cutoff.calc}}\cr
#' \code{\link{reg}}\cr
#' \code{\link{wcls}}\cr
#' \code{\link{cvGrid}}\cr
#' \code{\link{cvKfold}}\cr
#' \code{\link{workflow}}\cr
#'
#' @export
ensemble <- function(object,
                     n.ensemble = 100,
                     cutoff = 0.5,
                     # machine,
                     esCtrl,
                     unbalance.trt = c("None", "Over", "Under", "Balance")[1],
                     prop.keep = 1,
                     best.opti = TRUE,
                     min.keep = 10,
                     parallel = FALSE,
                     n.nodes = 4) {

  OOB.error <- list()
  cutoff.list <- c()
  model.list <- list()
  model.type <- ifelse(class(object) == "wClsObj", "Binary", "Continuous")
  performances <- data.frame()
  pred.table <- matrix(NA, n.ensemble, length(object@Y))
  loss <- numeric(length(n.ensemble))
  keep <- rep(TRUE, n.ensemble)
  min.keep <- max(min(min.keep, n.ensemble), 1)

  pb <- utils::txtProgressBar(min = 0, max = n.ensemble, style = 3)
  if(parallel) {

    browser()
    cl <- parallel::makeCluster(n.nodes)
    parallel::clusterExport(cl, list("splitData", "getSplit"))
    doParallel::registerDoParallel(cl)

    result <- foreach::foreach(i = 1:n.ensemble) %dopar% {

      if(class(object) != "RegObj" && unbalance.trt != "None") {

        y.table <- table(object@Y)
        class.size <- ifelse(unbalance.trt == "Over", max(y.table),
                             ifelse(unbalance.trt == "Under", min(y.table),
                                    ifelse(unbalance.trt == "Balance", mean(y.table), stop("Invalid input for Unbalance Treatment. "))))
        train.ind2 <- c()
        for(k in 1:length(unique(object@Y)))
          train.ind2 <- c(train.ind2, sample(which(object@Y == names(y.table)[k]), class.size, replace = TRUE))

        train.boot <- splitData(object, train.ind2)
      } else
        train.boot <- splitData(object, "bootstrap")

      trainObj <- train.boot$trainObj
      validObj <- train.boot$validObj
      trainObj.ind <- train.boot$split
      args <- removeArg(esCtrl, "machine")
      args <- appendArg(args, "valid", validObj, 1)

      if(esCtrl$machine != "cvGrid")
        model <- do.call(esCtrl$machine, appendArg(args, "object", trainObj, 1))
      else {

        args <- appendArg(args, "progress", FALSE, 0)
        model <- do.call(esCtrl$machine, appendArg(args, "trainObj", trainObj, 1))@best.model
      }

      return.mod <- model@model

      pred <- predict(model, object, cutoff = cutoff)
      if(class(object) == "RegObj") {

        return.pred <- pred@pred
        return.loss <- mean((validObj@Y - mean(validObj@Y))**2) - mean((validObj@Y - pred@pred[-trainObj.ind])**2)

      } else {

        return.pred <- pred@prob[, 1]
        # loss[i] <- sum((pred@pred[-trainObj.ind] == validObj@Y)*validObj@weight) / sum((validObj@weight))
        return.loss <- sum((log(mean(validObj@Y == validObj@Y.lvl[1])) * (validObj@Y == validObj@Y.lvl[1]) -
                              log(1-mean(validObj@Y == validObj@Y.lvl[1])) * (1-(validObj@Y == validObj@Y.lvl[1])) +
                              log(pred@prob[-trainObj.ind, 1]) * (validObj@Y == validObj@Y.lvl[1]) +
                              log(1-pred@prob[-trainObj.ind, 1]) * (1-(validObj@Y == validObj@Y.lvl[1])))*validObj@weight) / sum((validObj@weight))
      }

      utils::setTxtProgressBar(pb, i)

      list(return.mod, return.pred, return.loss)
    }

    foreach::registerDoSEQ()
    parallel::stopCluster(cl)

    for(i in 1:n.ensemble) {

      model.list[[i]] <- result[[i]][[1]]
      pred.table[i, ] <- result[[i]][[2]]
      loss[i] <- result[[i]][[3]]
    }
  } else {

    for(i in 1:n.ensemble) {

      if(class(object) != "RegObj" && unbalance.trt != "None") {

        y.table <- table(object@Y)
        class.size <- ifelse(unbalance.trt == "Over", max(y.table),
                             ifelse(unbalance.trt == "Under", min(y.table),
                                    ifelse(unbalance.trt == "Balance", mean(y.table), stop("Invalid input for Unbalance Treatment. "))))
        train.ind2 <- c()
        for(k in 1:length(unique(object@Y)))
          train.ind2 <- c(train.ind2, sample(which(object@Y == names(y.table)[k]), class.size, replace = TRUE))

        train.boot <- splitData(object, train.ind2)
      } else
        train.boot <- splitData(object, "bootstrap")

      trainObj <- train.boot$trainObj
      validObj <- train.boot$validObj
      trainObj.ind <- train.boot$split
      args <- removeArg(esCtrl, "machine")
      args <- appendArg(args, "valid", validObj, 1)

      if(esCtrl$machine != "cvGrid")
        model <- do.call(esCtrl$machine, appendArg(args, "object", trainObj, 1))
      else {

        args <- appendArg(args, "progress", FALSE, 0)
        model <- do.call(esCtrl$machine, appendArg(args, "trainObj", trainObj, 1))@best.model
      }

      model.list[[i]] <- model@model

      pred <- predict(model, object, cutoff = cutoff)
      if(class(object) == "RegObj") {

        pred.table[i, ] <- pred@pred
        loss[i] <- mean((validObj@Y - mean(validObj@Y))**2) - mean((validObj@Y - pred@pred[-trainObj.ind])**2)

      } else {

        pred.table[i, ] <- pred@prob[, 1]
        # loss[i] <- sum((pred@pred[-trainObj.ind] == validObj@Y)*validObj@weight) / sum((validObj@weight))
        loss[i] <- sum((-log(mean(validObj@Y == validObj@Y.lvl[1])) * (validObj@Y == validObj@Y.lvl[1]) -
                          log(1-mean(validObj@Y == validObj@Y.lvl[1])) * (1-(validObj@Y == validObj@Y.lvl[1])) +
                          log(pred@prob[-trainObj.ind, 1]) * (validObj@Y == validObj@Y.lvl[1]) +
                          log(1-pred@prob[-trainObj.ind, 1]) * (1-(validObj@Y == validObj@Y.lvl[1])))*validObj@weight) / sum((validObj@weight))
      }

      utils::setTxtProgressBar(pb, i)
    }
  }
  close(pb)

  if(best.opti && class(object) == "RegObj") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(mean((object@Y - colMeans(pred.table[loss >= sort(loss)[i], ]))**2), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && class(object) == "wClsObj") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(-sum((object@Y == object@Y.lvl[1]) * log(colMeans(pred.table[loss >= sort(loss)[i], ])) +
                          (1-(object@Y == object@Y.lvl[1])) * log(1-colMeans(pred.table[loss >= sort(loss)[i], ]))*
                          object@weight)/sum(object@weight), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  }

  if(prop.keep[1] < 1)
    keep <- loss >= sort(loss)[min(max(floor(n.ensemble*(1-prop.keep[1])), 1), n.ensemble - min.keep + 1)]
  if(length(prop.keep) > 1)
    for(i in 2:length(prop.keep))
      keep <- rbind(keep, loss >= sort(loss)[min(max(floor(n.ensemble*(1-prop.keep[i])), 1), n.ensemble - min.keep + 1)])

  return(methods::new("ModelEnsembleObj",
                      model.list = model.list,
                      model.type = model.type,
                      loss = loss,
                      keep = keep))
}
