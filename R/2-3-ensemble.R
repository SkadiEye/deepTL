ensemble_dnnet <- function(object,
                           n.ensemble = 100,
                           esCtrl,
                           unbalance.trt = c("None", "Over", "Under", "Balance")[1],
                           prop.keep = 1,
                           best.opti = TRUE,
                           min.keep = 10) {

  OOB.error <- list()
  model.list <- list()
  model.type <- ifelse(class(object@y) == "factor", "Binary", "Continuous")
  pred.table <- matrix(NA, n.ensemble, length(object@y))
  loss <- numeric(length(n.ensemble))
  keep <- rep(TRUE, n.ensemble)
  min.keep <- max(min(min.keep, n.ensemble), 1)

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

    pred <- predict(model, object@x, cutoff = cutoff)
    if(model.type == "Continuous") {

      pred.table[i, ] <- pred
      loss[i] <- sum(((validObj@y - mean(validObj@y))**2 - (validObj@y - pred[-trainObj.ind])**2) *
                       validObj@w) / sum(validObj@w)

    } else {

      pred.table[i, ] <- pred[, levels(validObj@y)[1]]
      loss[i] <- sum((-log(mean(validObj@y == levels(validObj@y)[1])) * (validObj@y == levels(validObj@y)[1]) -
                        log(1-mean(validObj@y == levels(validObj@y)[1])) * (1-(validObj@y == levels(validObj@y)[1])) +
                        log(pred[-trainObj.ind, levels(validObj@y)[1]]) * (validObj@y == levels(validObj@y)[1]) +
                        log(1-pred[-trainObj.ind, levels(validObj@y)[1]]) * (1-(validObj@y == levels(validObj@y)[1]))) *
                       validObj@w) / sum((validObj@w))
    }

    utils::setTxtProgressBar(pb, i)
  }
  close(pb)

  if(best.opti && model.type == "Continuous") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(sum((object@y - colMeans(pred.table[loss >= sort(loss)[i], ]))**2 *
                          object@w) / sum(object@w), Inf)

    keep <- loss >= sort(loss)[which.min(mse[1:(n.ensemble - min.keep + 1)])]
  } else if(best.opti && model.type == "Binary") {

    mse <- numeric(n.ensemble - 1)
    for(i in 1:length(mse))
      mse[i] <- min(-sum((object@y == levels(validObj@y)[1]) * log(colMeans(pred.table[loss >= sort(loss)[i], ])) +
                           (1-(object@y == levels(validObj@y)[1])) * log(1-colMeans(pred.table[loss >= sort(loss)[i], ])) *
                           object@w) / sum(object@w), Inf)

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
                      loss = loss,
                      keep = keep))
}
