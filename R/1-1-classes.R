#################################
#### dnnetInput class
#' An S4 class containing predictors (x), response (y) and sample weights (w)
#'
#' @slot x A numeric matrix, the predictors
#' @slot y A factor or numeric vector, either the class labels or continuous responses
#' @slot w A numeric vector, sample weights
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' @export
setClass("dnnetInput",
         slots = list(
           x = "matrix",
           y = "ANY",
           w = "numeric"
         ))

#################################
#### dnnetSurvInput class
#' An S4 class containing predictors (x), censoring time (y),
#' sample weights (w) and event status (e)
#'
#' @slot x A numeric matrix, the predictors
#' @slot y A numeric vector, censoring time
#' @slot w A numeric vector, sample weights
#' @slot e A numeric vector, event status
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' @export
setClass("dnnetSurvInput",
         slots = list(
           x = "matrix",
           y = "numeric",
           w = "numeric",
           e = "numeric"
         ))

#################################
#### dnnet class
#' An S4 class containing a deep neural network
#'
#' @slot norm A list, containing the centers and s.d.'s of x matrix and y vector (if numeric)
#' @slot weight A list of matrices, weight matrices in the fitted neural network
#' @slot bias A list of vectors, bias vectors in the fitted neural network
#' @slot loss The minimum loss acheived from the validate set
#' @slot loss.traj The loss trajectory
#' @slot label If the model is classification, a character vectors containing class labels
#' @slot model.type Either "Classification" or "Regression"
#' @slot model.spec Other possible information
#'
#' @seealso
#' \code{\link{dnnetInput-class}}\cr
#' @export
setClass("dnnet",
         slots = list(
           norm = "list",
           weight = "list",
           bias = "list",
           loss = "numeric",
           loss.traj = "numeric",
           label = "character",
           model.type = "character",
           model.spec = "list"
         ))

#################################
#### trtInput class
#' An S4 class containing predictors (x), response (y) and treatment assignment (z)
#'
#' @slot x A numeric matrix, the predictors
#' @slot y A factor or numeric vector, either the class labels or continuous responses
#' @slot z Treatment assignment
#'
#' @export
setClass("trtInput",
         slots = list(
           x = "matrix",
           y = "numeric",
           z = "ANY"
         ))

#################################
#### dnnetEnsemble class
#' An S4 class containing an ensemble of deep neural networks
#'
#' @slot model.list A list of dnnet models
#' @slot model.type Either "Classification" or "Regression"
#' @slot model.spec Other possible information
#' @slot loss A numeric vector of loss from all DNNs
#' @slot keep Whether the model is kept when put in the predict function
#'
#' @seealso
#' \code{\link{dnnet-class}}\cr
#' @export
setClass("dnnetEnsemble",
         slots = list(
           model.list = "list",
           model.type = "character",
           model.spec = "list",
           loss = "numeric",
           keep = "ANY"
         ))
