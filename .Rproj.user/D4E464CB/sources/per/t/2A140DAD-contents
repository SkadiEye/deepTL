#' @rdname dwnnel-predict
#' @section Methods (by signature):
#' \code{ModelObj}: newData should be a \code{TrtDataObj}, \code{wClsObj} or \code{RegObj} object. And a \code{PredDiscreteObj} will
#'  be returned for a classification model or a \code{PredContinuousObj} will be returned for a regression model.
#' @export
setMethod("predict", "ModelObj",
          function(object, newData, cutoff = 0.5) {

            if(!class(newData) %in% c("TrtDataObj", "wClsObj", "RegObj"))
              stop("Please import data as a data object before predicting. ")
            # if(class(newData) != paste("Train", object@model.type, "Obj", sep=''))
            #   stop("Model type must match test set type. ")

            # v.index <- match(object@v.names, newData@v.names)
            # if (sum(is.na(v.index)))
            #   stop("The test dataset does not include all variables in the model. ")
            # test <- newData@X[, v.index]
            test <- as.matrix(newData@X)
            # colnames(test) <- object@v.names

            # v.score <- data.frame(v.names = object@v.names, score = 1)
            v.score <- data.frame(v.names = colnames(newData@X), score = 1)

            if(class(newData) %in% c("wClsObj")) {

              if("svm" %in% class(object@model)) {

                prob <- attr(predict(object@model, test, probability = TRUE), "probabilities")
                label <- colnames(prob)

              } else if("randomForest" %in% class(object@model)) {

                # print(dim(test))
                colnames(test) <- NULL
                prob <- unclass(predict(object@model, test, type = "prob"))
                label <- colnames(prob)

                rf.imp <- object@model$importance
                v.score$score <- 0
                v.score$score[match(rownames(rf.imp), v.score$v.names)] <- rf.imp[, match("MeanDecreaseGini", colnames(rf.imp))] /
                  sum(rf.imp[, match("MeanDecreaseGini", colnames(rf.imp))])

              } else if("glmnet" %in% class(object@model)) {

                prob <- predict(object@model, test, type='response')
                label <- object@model$classnames
                if (class(newData) == "wClsObj")
                  prob <- cbind(1-prob, prob)
                else
                  prob <- prob[, , 1]

                if(class(newData) == "wClsObj") {

                  beta <- as.matrix(object@model$beta)
                  v.score$score[-match(rownames(beta)[beta[, 1] != 0], v.score$v.names)] <- 0
                } else {

                  beta <- as.data.frame(lapply(object@model$beta, function(x) as.matrix(x)))
                  v.score$score[-match(rownames(beta)[rowSums(beta != 0) > 0], v.score$v.names)] <- 0
                }

              } else if("gbm" %in% class(object@model)) {

                # if(class(newData) != "TrainBinaryObj")
                #   stop("Cannot perform GBM on MultiClass Response. ")

                if(object@model$cv.folds <= 1)
                  prob <- predict(object@model, data.frame(test, check.names = FALSE), n.trees = object@model$n.trees, type = "response")
                else
                  prob <- predict(object@model, data.frame(test, check.names = FALSE), type = "response")
                if(class(newData) == "wClsObj") {

                  prob <- cbind(prob, 1-prob)
                  label <- object@model$unique.responses
                } else {

                  prob <- prob[, , 1]
                  label <- colnames(prob)
                }

                gbm.imp <- summary(object@model, plotit = FALSE)
                v.score$score <- 0
                v.score$score[match(gsub("^\`", "",gsub("\`$", "", rownames(gbm.imp))), v.score$v.names)] <-
                  gbm.imp$rel.inf / sum(gbm.imp$rel.inf)

              } else if("nnet" %in% class(object@model)) {

                prob <- predict(object@model, test, type="raw")
                label <- colnames(prob)

              } else if("h2o" %in% attr(class(object@model), "package")) {

                #### NEED TO UPDATE JAVE
              } else if("lda" %in% class(object@model) || "qda" %in% class(object@model)) {

                prob <- predict(object@model, data.frame(test, check.names = FALSE))$posterior
                label <- object@model$lev

              } else if("pamrtrained" %in% class(object@model)) {

                prob <- pamr::pamr.predict(object@model, t(test), threshold = object@model$threshold, type = "posterior")
                label <- colnames(prob)

              } else if("dnnet" %in% class(object@model)) {

                prob <- predict(object@model, test)
                label <- colnames(prob)

              } else if("wkSVMObj" %in% class(object@model) || "wlSVMObj" %in% class(object@model)) {

                prob <- predict(object@model, test)
                label <- colnames(prob)

              }

              if (class(newData) == "wClsObj" && match(newData@Y.lvl[1], label) == 2) {
                prob <- prob[, 2:1]
                label <- label[2:1]
              }

              if (class(newData) == "wClsObj")
                pred <- factor(label[2 - (prob[, 1] > cutoff)], levels = label)
              else
                pred <- factor(label[apply(prob, 1, which.max)], levels = label)

              colnames(prob) <- label
              predObj <- methods::new("PredDiscreteObj",
                                      v.score = v.score,
                                      prob = prob,
                                      pred = pred)
            } else {

              if("glmnet" %in% class(object@model)) {

                pred <- predict(object@model, test)[, 1]

                beta <- as.matrix(object@model$beta)
                v.score$score[-match(rownames(beta)[beta[, 1] != 0], v.score$v.names)] <- 0

              } else if("svm" %in% class(object@model)) {

                pred <- predict(object@model, test)

              } else if("randomForest" %in% class(object@model)) {

                # print(dim(test))
                colnames(test) <- NULL
                pred <- predict(object@model, test)

                rf.imp <- object@model$importance
                v.score$score <- 0
                v.score$score[match(rownames(rf.imp), v.score$v.names)] <- rf.imp[, match("IncNodePurity", colnames(rf.imp))] /
                  sum(rf.imp[, match("IncNodePurity", colnames(rf.imp))])

              } else if("gbm" %in% class(object@model)) {

                if(object@model$cv.folds <= 1)
                  pred <- predict(object@model, data.frame(test, check.names = FALSE), n.trees = object@model$n.trees)
                else
                  pred <- predict(object@model, data.frame(test, check.names = FALSE))

                gbm.imp <- summary(object@model, plotit = FALSE)
                v.score$score <- 0
                v.score$score[match(gsub("^\`", "",gsub("\`$", "", rownames(gbm.imp))), v.score$v.names)] <-
                  gbm.imp$rel.inf / sum(gbm.imp$rel.inf)

              } else if("nnet" %in% class(object@model)) {

                pred <- predict(object@model, test)[, 1]

              } else if("dnnet" %in% class(object@model)) {

                pred <- predict(object@model, test)
              }

              predObj <- methods::new("PredContinuousObj",
                                      v.score = v.score,
                                      pred = pred)
            }

            return(predObj)
          }
)

#' @rdname dwnnel-predict
#' @section Methods (by signature):
#' \code{ModelEnsembleObj}: newData should be a \code{TrtDataObj}, \code{wClsObj} or \code{RegObj} object.
#'  And a \code{PredDiscreteObj} will
#'  be returned for a classification model or a \code{PredContinuousObj} will be returned for a regression model.
#'  If this \code{ModelEnsembleObj} indicates different combinations of model to keep, a
#'  \code{ListPredObj} will be returned.
#' @export
setMethod("predict", "ModelEnsembleObj",
          function(object, newData, cutoff = 0.5) {

            pred.table <- matrix(NA, nrow(newData@X), length(object@model.list))
            v.score <- data.frame(v.names = colnames(newData@X), score = 0)
            if(class(object@keep) == "matrix") calc <- colSums(object@keep) > 0
            else calc <- object@keep

            for(i in 1:length(object@model.list)) {

              if(calc[i]) {

                pred <- predict(methods::new("ModelObj",
                                             model = object@model.list[[i]],
                                             model.type = object@model.type),
                                newData = newData,
                                cutoff = cutoff)

                v.score$score[match(pred@v.score$v.names, v.score$v.names)] <-
                  v.score$score[match(pred@v.score$v.names, v.score$v.names)] + pred@v.score$score

                if(object@model.type == "Binary")
                  pred.table[, i] <- pred@prob[, 1]
                else if(object@model.type == "Continuous")
                  pred.table[, i] <- pred@pred
                else {

                  if(i == min(which(calc)))
                    pred.table <- array(NA, dim = c(nrow(pred@prob), ncol(pred@prob), length(object@model.list)))

                  pred.table[, , i] <- pred@prob
                }

                # if(i == min(which(calc)) && object@model.type != "Continuous")
                if(object@model.type != "Continuous")
                  label <- colnames(pred@prob)
              }
            }

            # v.score$score <- v.score$score/length(object@model.list)
            if(class(object@keep) != "matrix") {

              v.score$score <- v.score$score/sum(calc)

              if(object@model.type == "Continuous")
                return(methods::new("PredContinuousObj", v.score = v.score, pred = rowMeans(pred.table, na.rm = TRUE)))

              else {

                if(object@model.type == "Binary") {

                  prob <- rowMeans(pred.table, na.rm = TRUE)
                  prob <- as.matrix(cbind(prob, 1-prob))
                  colnames(prob) <- label
                  pred <- factor(label[2 - (prob[, 1] > cutoff)], levels = label)
                } else {

                  prob <- apply(pred.table, 1:2, mean)
                  colnames(prob) <- label
                  pred <- factor(label[apply(prob, 1, which.max)], levels = label)
                }

                return(methods::new("PredDiscreteObj",
                                    v.score = v.score,
                                    prob = prob,
                                    pred = pred))
              }
            } else {

              list.pred <- list()
              for(i in 1:dim(object@keep)[1]) {

                v.score$score <- v.score$score/sum(calc)

                if(object@model.type == "Continuous")
                  list.pred[[i]] = methods::new("PredContinuousObj", v.score = v.score, pred = rowMeans(pred.table[, object@keep[i, ]], na.rm = TRUE))

                else {

                  if(object@model.type == "Binary") {

                    prob <- rowMeans(pred.table[, object@keep[i, ]], na.rm = TRUE)
                    prob <- as.matrix(cbind(prob, 1-prob))
                    colnames(prob) <- label
                    pred <- factor(label[2 - (prob[, 1] > cutoff)], levels = label)
                  } else {

                    prob <- apply(pred.table[, , object@keep[i, ]], 1:2, mean)
                    colnames(prob) <- label
                    pred <- factor(label[apply(prob, 1, which.max)], levels = label)
                  }

                  list.pred[[i]] <- methods::new("PredDiscreteObj", v.score = v.score, prob = prob, pred = pred)
                }
              }
              return(methods::new("ListPredObj", listPred = list.pred))
            }
          }
)

#' @rdname dwnnel-predict
#' @section Methods (by signature):
#' \code{cvGrid}: The best model selected by grid search will be used. The rest should be the same as
#'  \code{ModelObj}.
#' @export
setMethod("predict", "cvGrid",
          function(object, newData, cutoff = 0.5) {

            predict(object@best.model, newData = newData, cutoff = cutoff)
          }
)

#' @rdname dwnnel-predict
#' @section Methods (by signature):
#' \code{ITRObj}: newData should be a \code{TrtDataObj} object. And a \code{PredDiscreteObj} will
#'  be returned.
#' @export
setMethod("predict", "ITRObj",
          function(object, newData, cutoff = 0.5) {

            if(object@model.type %in% c("OWL", "DOWL")) {

              newData <- methods::new("wClsObj", X = newData@X, Y = newData@trtLabl, Y.lvl = newData@trtLevl)
              return(predict(object@model, newData = newData, cutoff = cutoff))

            } else if(object@model.type == "VT") {

              newData <- methods::new("RegObj", X = newData@X)
              pred1 <- predict(object@model[[1]], newData = newData)@pred
              pred2 <- predict(object@model[[2]], newData = newData)@pred
              prob <- cbind(exp(pred1), exp(pred2))/(exp(pred1) + exp(pred2))
              label <- object@label
              colnames(prob) <- label
              pred <- factor(label[2 - (prob[, 1] > cutoff)], levels = label)
              return(methods::new("PredDiscreteObj", prob = prob, pred = pred))

            } else if(object@model.type == "Simple") {

              newData <- methods::new("RegObj", X = newData@X)
              pred <- predict(object@model, newData = newData)@pred
              prob <- cbind(exp(pred), 1)/(exp(pred) + 1)
              label <- object@label
              colnames(prob) <- label
              pred <- factor(label[2 - (prob[, 1] > cutoff)], levels = label)
              return(methods::new("PredDiscreteObj", prob = prob, pred = pred))

            } else if(object@model.type == "VTS") {

              prob <- predict(object@model, newData@X, type = "prob")[, 2:1]
              label <- object@label
              colnames(prob) <- label
              pred <- factor(label[2 - (prob[, 1] > cutoff)], levels = label)
              return(methods::new("PredDiscreteObj", prob = prob, pred = pred))
            }
          }
)

############################################
#### Calculate the cutoff

#' Cutoff Calculation for binary class predictions
#'
#' Cutoff Calculation for binary class predictions via different ways.
#'
#' @param cutoff If it's a number between 0 and 1, return this number.
#' If cutoff == 'prevalence', the prevalence of the first class will be returned.
#' if cutoff == 'optimal', the cutoff will be optimized by maximizing AUC using the training set.
#' @param trainObj A \code{wClsObj} object.
#' @param modelObj A \code{wcls} object.
#'
#' @return A \code{numeric} value as the cutoff
#'
#' @export
cutoff.calc <- function(cutoff, trainObj = NULL, modelObj = NULL) {

  if(is.numeric(cutoff))
    return(cutoff)

  if(cutoff == "prevalence")
    return(mean(trainObj@Y == trainObj@Y.lvl[1]))

  if(cutoff == "optimal"){

    pred <- predict(modelObj, trainObj)
    prediction <- ROCR::prediction(pred@prob[, 1], trainObj@Y == trainObj@Y.lvl[1])
    perf <- ROCR::performance(prediction, "tpr", "fpr")
    return(min(1, max(0, prediction@cutoffs[[1]][which.min(perf@x.values[[1]]**2 + (1-perf@y.values[[1]])**2)])))
  }

  stop("Cutoff is either numeric, `prevalence` or `optimal`. ")
}

#########################################################################33
#### Evaluate the Performance of the Prediction

#' @name eval
#' @rdname eval
#'
#' @title Evaluate the Performance of the Prediction
#'
#' @description Evaluate the performance of the prediction using different criteria.
#'
#' @details
#'
#' acc: Accuracy
#' sens: sensitivity
#' spec: specificity
#' auc: area under the curve
#' ppv: positive predictive value
#' npv: negative predictive value
#' w.acc: weighted accuracy --- w.acc = sum((Y==pred.Y)*weight)/sum(weight)
#' a.acc: thresholded accuracy --- a.acc = sum((Y==pred.Y)*inclusion)/sum(inclusion)
#' val.f: value function  --- val.f = sum((Y.val==pred.Y)*weight)/sum(weight)
#'
#' @param object A \code{PredObj} or a \code{ListPredObj} object.
#' @param test.Y The true response.
#' @param cutoff A cutoff for binary response.
#' @param type The type of the response, either "Binary" or "Continuous".
#' @param output The output types. See details.
#' @param weight Sample weight for weighted accuracy.
#' @param inclusion Used in thresholded accuracy.
#' @param Y.val Used in the value function.
#'
#' @return A \code{data.frame} of performance will be retured.
#'
#' @seealso
#' \code{\link{dwnnel-predict}}
NULL

#' @rdname eval
#' @export
setGeneric("evalPred",
           function(object, test.Y, cutoff = 0.5,
                    type = c("Binary", "Continuous")[1],
                    output = ifelse(type == "Continuous",
                                    list("mse"),
                                    list(c("acc", "sens", "spec", "auc", "ppv", "npv", "w.acc", "a.acc", "val.f")))[[1]],
                    weight = rep(1, length(test.Y)), inclusion = rep(TRUE, length(test.Y)),
                    Y.val = test.Y,
                    Y.resp = rep(1, length(test.Y))) standardGeneric("evalPred")
)

#' @rdname eval
#' @export
setMethod("evalPred", "PredObj",
          function(object, test.Y, cutoff, type, output, weight, inclusion, Y.val, Y.resp) {

            if(class(object) == "PredDiscreteObj" && type == "Continuous")
              stop("Please match the type of the prediction and the type of the response. ")
            if(class(object) == "PredContinuousObj" && type != "Continuous")
              stop("Please match the type of the prediction and the type of the response. ")
            if(class(object) == "PredDiscreteObj" && "mse" %in% output)
              stop("Cannot calculate MSE for a discrete response. ")
            if(class(object) == "PredContinuousObj" &&
               sum(c("acc", "sens", "spec", "auc", "ppv", "npv", "w.acc", "a.acc", "val.f") %in% output))
              stop("Cannot calculate acc, sens, spec or auc for a continuous response. ")
            if(sum(is.na(match(output, c("acc", "sens", "spec", "auc", "ppv", "npv", "mse", "w.acc", "a.acc", "val.f")))))
              stop("Output statistics out of bound. ")

            if(type == "Continuous") {

              if(sum(is.na(object@pred)))
                return(data.frame(mse = NA))

              return(data.frame(mse = sum((object@pred - test.Y)**2)/(length(test.Y) - 1)))
            } else {

              out.dat <- list()

              # if(type == "MultiClass")
              #   cutoff = 0.5
              if("acc" %in% output)
                out.dat[["acc"]] <- mean(object@pred == test.Y)
              if("w.acc" %in% output)
                out.dat[["w.acc"]] <- sum(weight*(object@pred == test.Y)) / sum(weight)
              if("a.acc" %in% output)
                out.dat[["a.acc"]] <- sum(inclusion*(object@pred == test.Y)) / sum(inclusion)
              if("val.f" %in% output)
                out.dat[["val.f"]] <- mean(Y.resp*(object@pred == Y.val))

              for(i in 1:ifelse(type == "Binary", 1, dim(object@prob)[2])) {

                if(length(unique(test.Y)) == 1) {

                  for(j in 1:length(output))
                    if(!output[j] %in% c("acc", "w.acc", "a.acc", "val.f"))
                      out.dat[[ifelse(type != "Binary",
                                      paste(output[j], colnames(object@prob)[i], sep='_'),
                                      output[j])]] <- 1/2
                } else {

                  # print(summary(object@prob[, i]))
                  # print(table(test.Y))
                  # print(colnames(object@prob))

                  try(prediction <- ROCR::prediction(object@prob[, i], as.numeric(test.Y == colnames(object@prob)[i])))
                  try(index <- max(which(prediction@cutoffs[[1]] > cutoff)))

                  for(j in 1:length(output)) {

                    if(exists("index")) {

                      if(!output[j] %in% c("acc", "w.acc", "a.acc", "val.f"))
                        out.dat[[ifelse(type != "Binary",
                                        paste(output[j], colnames(object@prob)[i], sep='_'),
                                        output[j])]] <-
                          ROCR::performance(prediction, output[j])@y.values[[1]][ifelse(output[j] == "auc", 1, index)]
                    } else {

                      if(!output[j] %in% c("acc", "w.acc", "a.acc", "val.f"))
                        out.dat[[ifelse(type != "Binary", paste(output[j], colnames(object@prob)[i], sep='_'), output[j])]] <- NA
                    }
                  }
                }
              }

              return(as.data.frame(out.dat))
            }

          }
)

#' @rdname eval
#' @export
setMethod("evalPred", "ListPredObj",
          function(object, test.Y, cutoff, type, output, weight, inclusion, Y.val) {

            performances <- data.frame()
            for(i in 1:length(object@listPred)) {

              performances <- rbind(performances,
                                    evalPred(object@listPred[[i]],
                                             test.Y, cutoff, type, output, weight, inclusion, Y.val))
            }

            return(performances)
          }
)


