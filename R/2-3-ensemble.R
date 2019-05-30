ensemble_dnnet <- function(object,
                           n.ensemble = 100,
                           esCtrl,
                           unbalance.trt = c("None", "Over", "Under", "Balance")[1],
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
    } else
      train.boot <- splitDnnet(object, "bootstrap")

    trainObj <- train.boot$train
    validObj <- train.boot$valid
    trainObj.ind <- train.boot$split
    args <- esCtrl # removeArg(esCtrl, "machine")
    args <- appendArg(args, "validate", validObj, 1)
    args <- appendArg(args, "train", trainObj, 1)

    model <- do.call(deepTL::dnnet, args)
    model.list[[i]] <- model
    if(i == 1) model.type <- model@model.type
    if(i == 1) model.spec <- model@model.spec

    pred <- predict(model, object@x, cutoff = cutoff)
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
      loss[i] <- -sum((mat_y[-trainObj.ind, ] * log(pred[-trainObj.ind, ]) -
                         mat_y[-trainObj.ind, ] * (rep(1, length(validObj@y)) %*% t(log(colMeans(mat_y[-trainObj.ind, ]))))) *
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
