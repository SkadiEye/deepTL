#' #### ##################################################################
#' #### #### Return the full name of an ITR framework
#'
#' #### #' Return the full name of an ITR framework
#' #### #'
#' #### #' Return the full name of an Individualized Treatment Rule (ITR) framework
#' #### #'
#' #### #' @param ITR A \code{character} of the abbreviation
#' #### #'
#' itr.names <- function(ITR) {
#'
#'   if(ITR == "itrDOWL") return("Doubly Outcome Weighted Learning")
#'   if(ITR == "itrOWL") return("Outcome Weighted Learning")
#'   if(ITR == "itrVT") return("Virtual Twins")
#'   if(ITR == "itrSimple") return("A simple Method")
#'   if(ITR == "EndLot") return("ENsemble Deep Learning Optimal Treatment")
#'   if(ITR == "LASSO") return("LASSO")
#'   if(ITR == "OWL_LASSO") return("Outcome Weighted Learning LASSO")
#'   if(ITR == "VT_Single") return("Virtual Twins (Single)")
#'   if(ITR == "VT_Ensemble") return("Virtual Twins (Ensemble)")
#' }
#'
#' ctrl.names <- function(ctrl) {
#'
#'   if(ctrl$machine == "ensemble") {
#'
#'     if(ctrl$esCtrl$machine == "cvGrid")
#'       return(paste(machine.names(ctrl$esCtrl$gsCtrl$machine), " (ensemble; grid search)"))
#'
#'     return(paste(machine.names(ctrl$esCtrl$machine), " (ensemble)"))
#'   }
#'   if(ctrl$machine == "cvGrid")
#'     return(paste(machine.names(ctrl$gsCtrl$machine), " (grid search)"))
#'
#'   return(machine.names(ctrl$machine))
#' }
#'
#' machine.names <- function(machine) {
#'
#'   if(machine %in% c("regSVR"))                return("SVR")
#'   if(machine %in% c("regANN", "wclsANN"))     return("ANN")
#'   if(machine %in% c("regDNN", "wclsDNN"))     return("DNN")
#'   if(machine %in% c("regLASSO", "wclsLASSO")) return("LASSO")
#'   if(machine %in% c("regRF"))                 return("RF")
#'   if(machine %in% c("wclskSVM"))              return("SVM")
#'   if(machine %in% c("wclslSVM"))              return("SVM")
#' }
#'
#' machine.full.names <- function(machine) {
#'
#'   if(machine %in% c("regSVR")) return("Support Vector Regression")
#'   if(machine %in% c("regANN", "wclsANN")) return("Artificial Neural Network")
#'   if(machine %in% c("regDNN", "wclsDNN")) return("Deep Neural Network")
#'   if(machine %in% c("regLASSO", "wclsLASSO")) return("LASSO")
#'   if(machine %in% c("regRF")) return("Random Forest")
#'   if(machine %in% c("wclskSVM")) return("Kernel Support Vector Machine")
#'   if(machine %in% c("wclslSVM")) return("Linear Support Vector Machine")
#' }
#'
#' ###########################################################
#' ### Individualized Treatment Rule Frameworks
#'
#' #' @name itr
#' #' @rdname itr
#' #'
#' #' @title Individualized Treatment Rule Frameworks
#' #'
#' #' @description A collection of Individualized Treatment Rule frameworks.
#' #'
#' #' @details
#' #'
#' #' These \code{itr} frameworks all works for the personalized treatment problems with
#' #'  two treatmnet labels. Different frameworks would call either regression methods
#' #'  or classification methods, or both. Methods of the same type could be applied in the
#' #'  same framework.
#' #'
#' #' @inheritParams itr
#' #' @param object A \code{TrtDataObj} object, used as the training set.
#' #' @param valid A \code{TrtDataObj} object, used as the validation set. The default is \code{NULL}.
#' #' @param regCtrl A \code{list} of arguments that specifies a regression method, handled by
#' #'  \code{\link{regCtrlPanel}}.
#' #' @param wclsCtrl A \code{list} of arguments that specifies a weighted classification method, handled by
#' #'  \code{\link{wclsCtrlPanel}}.
#' #' @param ... Other parameters passed to \code{itr} functions.
#' #'
#' #' @return Returns a \code{ITRObj} object.
#' #'
#' #' @seealso
#' #' \code{\link{TrtDataObj-class}}\cr
#' #' \code{\link{ITRObj-class}}\cr
#' #' \code{\link{regCtrlPanel}}\cr
#' #' \code{\link{wclsCtrlPanel}}\cr
#' #' \code{\link{reg}}\cr
#' #' \code{\link{wcls}}\cr
#' #' \code{\link{cvGrid}}\cr
#' #' \code{\link{ensemble}}\cr
#' #' \code{\link{cvKfold}}\cr
#' #' \code{\link{workflow}}\cr
#' #'
#' #' @references
#' #' \href{https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3636816/pdf/nihms379905.pdf}{Outcome Weighted Learning}\cr
#' #' \href{http://onlinelibrary.wiley.com/doi/10.1002/sim.4322/epdf}{Virtual Twins}\cr
#' #' \href{https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4338439/pdf/nihms637476.pdf}{A simple Regression}\cr
#' NULL
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("itrDOWL",
#'            function(object, valid = NULL, regCtrl, wclsCtrl, ...) standardGeneric("itrDOWL")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("itrOWL",
#'            function(object, valid = NULL, wclsCtrl, ...) standardGeneric("itrOWL")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("itrVT",
#'            function(object, valid = NULL, regCtrl, ...) standardGeneric("itrVT")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("itrSimple",
#'            function(object, valid = NULL, regCtrl, ...) standardGeneric("itrSimple")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("EndLot",
#'            function(object, valid = NULL, ...) standardGeneric("EndLot")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("LASSO",
#'            function(object, valid = NULL, ...) standardGeneric("LASSO")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("OWL_LASSO",
#'            function(object, valid = NULL, ...) standardGeneric("OWL_LASSO")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("VT_Single",
#'            function(object, valid = NULL, ...) standardGeneric("VT_Single")
#' )
#'
#' #' @rdname itr
#' #' @export
#' setGeneric("VT_Ensemble",
#'            function(object, valid = NULL, ...) standardGeneric("VT_Ensemble")
#' )
#'
#' #' @rdname itr
#' #' @section Methods (by generic):
#' #' \code{itrDOWL}: Doubly Weighted Outcome Learning Framework. This procedure first calls a
#' #'  regression model to estimate E(Y|X), the expected treatment effect averaged on all
#' #'  treatments. Then it calls a weighted classification to estimate the ITR.
#' #' @export
#' setMethod(
#'   "itrDOWL",
#'   "TrtDataObj",
#'   function(object, valid, regCtrl, wclsCtrl, ...) {
#'
#'     cat("  Reg:  ", ctrl.names(regCtrl), "\n")
#'
#'     regObj <- methods::new("RegObj", Y = object@trtResp, X = object@X)
#'     regArg <- removeArg(regCtrl, "machine")
#'     if(regCtrl$machine != "cvGrid")
#'       regArg <- appendArg(regArg, "object", regObj, 1)
#'     else
#'       regArg <- appendArg(regArg, "trainObj", regObj, 1)
#'     if(!is.null(valid)) {
#'
#'       regObj.valid <- methods::new("RegObj", Y = valid@trtResp, X = valid@X)
#'       regArg <- appendArg(regArg, "valid", regObj.valid, 1)
#'     }
#'     regMod <- do.call(regCtrl$machine, regArg)
#'     regPred <- predict(regMod, regObj)
#'
#'     cat("  Cls:  ", ctrl.names(wclsCtrl), "\n")
#'
#'     weight <- abs(object@trtResp - regPred@pred)
#'     rev.labl <- factor(3-as.numeric(object@trtLabl))
#'     levels(rev.labl) <- levels(object@trtLabl)
#'     Y.f <- ifelse(object@trtResp > regPred@pred, object@trtLabl, rev.labl)
#'     Y.f <- factor(levels(object@trtLabl)[Y.f], levels = levels(object@trtLabl))
#'     wclsObj <- methods::new("wClsObj", weight = weight, Y = Y.f, X = object@X, Y.lvl = object@trtLevl) # , Y.true = object@trtTrue)
#'
#'     wclsArg <- removeArg(wclsCtrl, "machine")
#'     if(wclsCtrl$machine != "cvGrid")
#'       wclsArg <- appendArg(wclsArg, "object", wclsObj, 1)
#'     else {
#'
#'       wclsArg <- appendArg(wclsArg, "trainObj", wclsObj, 1)
#'       wclsArg <- appendArg(wclsArg, "weight", object@sample.weight, 1)
#'       wclsArg <- appendArg(wclsArg, "inclusion", object@sample.inclsn, 1)
#'     }
#'
#'     if(!is.null(valid)) {
#'
#'       regPred.valid <- predict(regMod, regObj.valid)
#'       weight <- abs(valid@trtResp - regPred.valid@pred)
#'       rev.labl <- factor(3-as.numeric(valid@trtLabl))
#'       levels(rev.labl) <- levels(valid@trtLabl)
#'       Y.f <- ifelse(valid@trtResp > regPred.valid@pred, valid@trtLabl, rev.labl)
#'       Y.f <- factor(levels(valid@trtLabl)[Y.f], levels = levels(valid@trtLabl))
#'
#'       wclsObj.valid <- methods::new("wClsObj", weight = weight, Y = Y.f, X = valid@X, Y.lvl = valid@trtLevl) # , Y.true = valid@trtTrue)
#'       wclsArg <- appendArg(wclsArg, "valid", wclsObj.valid, 1)
#'     }
#'
#'     methods::new("ITRObj", model = do.call(wclsCtrl$machine, wclsArg), model.type = "OWL", label = levels(object@trtLabl))
#'   }
#' )
#'
#' setMethod(
#'   "EndLot",
#'   "TrtDataObj",
#'   function(object, valid, norm.x = TRUE, regCtrl = NULL, wclsCtrl = NULL,
#'            n.batch = NULL, n.epoch = NULL, n.ensemble = NULL, n.hidden = NULL,
#'            l1.reg = 10**-4, l2.reg = 0, plot = FALSE, best.opti = TRUE,
#'            learning.rate.adaptive = "adagrad", activate = "elu", ...) {
#'
#'     p.dim <- dim(object@X)[2]
#'     if(is.null(n.ensemble))
#'       n.ensemble <- ifelse(p.dim > 200, 10, ifelse(p.dim <= 100, 200, 25))
#'     if(is.null(n.batch))
#'       n.batch <- ifelse(p.dim > 100, 10, 100)
#'     if(is.null(n.hidden))
#'       n.hidden <- c(30, 30, 30)
#'     if(is.null(n.epoch)) {
#'
#'       n.epoch <- ifelse(p.dim <= 100, 1000,
#'                         ifelse(p.dim <= 1000, 250,
#'                                ifelse(p.dim <= 2000, 500,
#'                                       ifelse(p.dim <= 5000, 1000))))
#'     }
#'     if(is.null(regCtrl)) {
#'
#'       regCtrl.dnn <- list(machine = "regDNN", norm.x = norm.x, n.batch = n.batch, n.epoch = n.epoch, activate = activate, accel = "rcpp",
#'                           l1.reg = l1.reg, l2.reg = l2.reg, plot = plot ,
#'                           learning.rate.adaptive = learning.rate.adaptive, early.stop.det = 100)
#'       regCtrl <- list(machine = "ensemble", n.ensemble = n.ensemble, best.opti = best.opti,
#'                       esCtrl = append(regCtrl.dnn, list(n.hidden = n.hidden)))
#'     }
#'     if(is.null(wclsCtrl)) {
#'
#'       wclsCtrl.dnn <- list(machine = "wclsDNN", norm.x = norm.x, n.batch = n.batch, n.epoch = n.epoch, activate = activate, accel = "rcpp",
#'                            l1.reg = l1.reg, l2.reg = l2.reg, plot = plot,
#'                            learning.rate.adaptive = learning.rate.adaptive, early.stop.det = 100)
#'       wclsCtrl <- list(machine = "ensemble", n.ensemble = n.ensemble, best.opti = best.opti,
#'                        esCtrl = append(wclsCtrl.dnn, list(n.hidden = n.hidden)))
#'     }
#'
#'     return(itrDOWL(object = object, valid = valid, regCtrl = regCtrl, wclsCtrl = wclsCtrl))
#'   }
#' )
#'
#' setMethod(
#'   "OWL_LASSO",
#'   "TrtDataObj",
#'   function(object, valid, regCtrl = NULL, wclsCtrl = NULL,
#'            k.fold = 5, lambda = 0.8**(0:40), ...) {
#'
#'     regCtrl  <- list(machine = "cvGrid", k.fold = k.fold, gsCtrl = list(machine = "regLASSO"))
#'     wclsCtrl <- list(machine = "cvGrid", k.fold = k.fold, gsCtrl = list(machine = "wclsLASSO", lambda = lambda))
#'     return(itrDOWL(object = object, valid = valid, regCtrl = regCtrl, wclsCtrl = wclsCtrl))
#'   }
#' )
#'
#' #' @rdname itr
#' #' @section Methods (by generic):
#' #' \code{itrOWL}: Outcome Weighted Learning Framework. This procedure directly calls a weighted classification
#' #'  to estimate the ITR by outcome weighted learning.
#' #' @export
#' setMethod(
#'   "itrOWL",
#'   "TrtDataObj",
#'   function(object, valid, wclsCtrl, ...) {
#'
#'     cat("  Cls:  ", ctrl.names(wclsCtrl), "\n")
#'
#'     min.w <- min(object@trtResp)
#'     max.w <- max(object@trtResp)
#'     if(!is.null(valid)) {
#'
#'       min.w <- min(min.w, valid@trtResp)
#'       max.w <- max(max.w, valid@trtResp)
#'     }
#'
#'     wclsObj <- methods::new("wClsObj", weight = object@trtResp - min.w + (max.w - min.w)*0.01,
#'                             Y = object@trtLabl, X = object@X, Y.lvl = object@trtLevl)
#'     wclsArg <- removeArg(wclsCtrl, "machine")
#'     if(wclsCtrl$machine != "cvGrid")
#'       wclsArg <- appendArg(wclsArg, "object", wclsObj, 1)
#'     else {
#'
#'       wclsArg <- appendArg(wclsArg, "trainObj", wclsObj, 1)
#'       wclsArg <- appendArg(wclsArg, "weight", object@sample.weight, 1)
#'       wclsArg <- appendArg(wclsArg, "inclusion", object@sample.inclsn, 1)
#'     }
#'
#'     if(!is.null(valid)) {
#'
#'       wclsObj.valid <- methods::new("wClsObj", weight = valid@trtResp - min.w + (max.w - min.w)*0.01,
#'                                     Y = valid@trtLabl, X = valid@X, Y.lvl = valid@trtLevl)
#'       wclsArg <- appendArg(wclsArg, "valid", wclsObj.valid, 1)
#'     }
#'
#'     methods::new("ITRObj", model = do.call(wclsCtrl$machine, wclsArg), model.type = "OWL", label = levels(object@trtLabl))
#'   }
#' )
#'
#' #' @rdname itr
#' #' @section Methods (by generic):
#' #' \code{itrVT}: Virture Twins Framework. This procedure calls the same method to fit two regression
#' #'  models for patients with either treatment. The ITR is determined by comparing the predictions
#' #'  with both models.
#' #' @export
#' setMethod(
#'   "itrVT",
#'   "TrtDataObj",
#'   function(object, valid, regCtrl, ...) {
#'
#'     split.trt <- splitData(object, which(object@trtLabl == object@trtLevl[1]))
#'     caseObj <- split.trt$trainObj
#'     ctrlObj <- split.trt$validObj
#'
#'     regObj.case <- methods::new("RegObj", Y = caseObj@trtResp, X = caseObj@X)
#'     regArg.case <- removeArg(regCtrl, "machine")
#'     if(regCtrl$machine != "cvGrid")
#'       regArg.case <- appendArg(regArg.case, "object", regObj.case, 1)
#'     else
#'       regArg.case <- appendArg(regArg.case, "trainObj", regObj.case, 1)
#'     regObj.ctrl <- methods::new("RegObj", Y = ctrlObj@trtResp, X = ctrlObj@X)
#'     regArg.ctrl <- removeArg(regCtrl, "machine")
#'     if(regCtrl$machine != "cvGrid")
#'       regArg.ctrl <- appendArg(regArg.ctrl, "object", regObj.ctrl, 1)
#'     else
#'       regArg.ctrl <- appendArg(regArg.ctrl, "trainObj", regObj.ctrl, 1)
#'
#'     if(!is.null(valid)) {
#'
#'       split.trt <- splitData(valid, which(valid@trtLabl == valid@trtLevl[1]))
#'       valid.caseObj <- split.trt$trainObj
#'       valid.ctrlObj <- split.trt$validObj
#'       regObj.case.valid <- methods::new("RegObj", Y = valid.caseObj@trtResp, X = valid.caseObj@X)
#'       regObj.ctrl.valid <- methods::new("RegObj", Y = valid.ctrlObj@trtResp, X = valid.ctrlObj@X)
#'       regArg.case <- appendArg(regArg.case, "valid", regObj.case.valid, 1)
#'       regArg.ctrl <- appendArg(regArg.ctrl, "valid", regObj.ctrl.valid, 1)
#'     }
#'
#'     cat("  Reg:  ", ctrl.names(regCtrl), "\n")
#'
#'     regMod.case <- do.call(regCtrl$machine, regArg.case)
#'
#'     cat("  Reg:  ", ctrl.names(regCtrl), "\n")
#'
#'     regMod.ctrl <- do.call(regCtrl$machine, regArg.ctrl)
#'
#'     methods::new("ITRObj", model = list(model.case = regMod.case, model.ctrl = regMod.ctrl), model.type = "VT", label = levels(object@trtLabl))
#'   }
#' )
#'
#' setMethod(
#'   "VT_Ensemble",
#'   "TrtDataObj",
#'   function(object, valid, ntree = 2000, ...) {
#'
#'     regCtrl <- list(machine = "regRF", ntree = ntree)
#'     return(itrVT(object = object, valid = valid, regCtrl = regCtrl))
#'   }
#' )
#'
#' setMethod(
#'   "VT_Single",
#'   "TrtDataObj",
#'   function(object, valid, ntree = 2000, k.fold = 5, cp_list = seq(0.01, 0.2, 0.01), ...) {
#'
#'     regCtrl <- list(machine = "regRF", ntree = ntree)
#'     itr_ <- itrVT(object = object, valid = valid, regCtrl = regCtrl)
#'     y1 <- predict(itr_@model$model.case@model, object@X)
#'     y0 <- predict(itr_@model$model.ctrl@model, object@X)
#'     y <- (y1 > y0)*1
#'     x <- object@X
#'
#'     n_sample <- dim(object@X)[1]
#'     resample <- sample(n_sample)
#'     result <- matrix(NA, n_sample, length(cp_list))
#'     accu <- numeric(length(cp_list))
#'
#'     for (j in 1:length(cp_list)) {
#'       for(i in 1:k.fold) {
#'
#'         ind <- resample[floor(n_sample*(i-1)/k.fold+1):floor(n_sample*i/k.fold)]
#'         tree_mod <- rpart::rpart(Y ~ ., data = data.frame(x[-ind, ], "Y" = y[-ind]), method = "class", cp = cp_list[j])
#'         result[ind, j] <- (predict(tree_mod, x[ind, ])[, 2] > 0.5)*1
#'       }
#'
#'       accu[j] <- mean(y == result[, j])
#'     }
#'
#'     cp_best <- cp_list[which.max(accu)]
#'     mod <- rpart::rpart(Y ~ ., data = data.frame(x, "Y" = y), method = "class", cp = cp_list[j])
#'
#'     methods::new("ITRObj", model = mod, model.type = "VTS", label = levels(object@trtLabl))
#'   }
#' )
#'
#' #' @rdname itr
#' #' @section Methods (by generic):
#' #' \code{itrSimple}: Simple Regression Framework. This procedure fits one regression model with
#' #'  A*Y ~ X, where A from {1, -1} is the treatment label, Y is the continuous outcome and X is the
#' #'  predictors.
#' #' @export
#' setMethod(
#'   "itrSimple",
#'   "TrtDataObj",
#'   function(object, valid, regCtrl, ...) {
#'
#'     cat("  Reg:  ", ctrl.names(regCtrl), "\n")
#'
#'     regObj <- methods::new("RegObj", Y = (object@trtResp - mean(object@trtResp)) * ((object@trtLabl == object@trtLevl[1])*2 - 1), X = object@X)
#'     regArg <- removeArg(regCtrl, "machine")
#'     if(regCtrl$machine != "cvGrid")
#'       regArg <- appendArg(regArg, "object", regObj, 1)
#'     else
#'       regArg <- appendArg(regArg, "trainObj", regObj, 1)
#'     if(!is.null(valid)) {
#'
#'       regObj.valid <- methods::new("RegObj", Y = valid@trtResp, X = valid@X)
#'       regArg <- appendArg(regArg, "valid", regObj.valid, 1)
#'     }
#'     methods::new("ITRObj", model = do.call(regCtrl$machine, regArg), model.type = "Simple", label = levels(object@trtLabl))
#'   }
#' )
#'
#' setMethod(
#'   "LASSO",
#'   "TrtDataObj",
#'   function(object, valid, k.fold = 5, ...) {
#'
#'     regCtrl  <- list(machine = "cvGrid", k.fold = k.fold, gsCtrl = list(machine = "regLASSO"))
#'     return(itrSimple(object = object, valid = valid, regCtrl = regCtrl))
#'   }
#' )
