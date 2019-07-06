###########################################################
### Define functions: importDnnet and splitDnnet

#' Import Data to create a \code{dnnetInput} object.
#'
#' @param x A \code{matrix} containing all samples/variables. It has to be \code{numeric}
#' and cannot be left blank. Any variable with missing value will be removed.
#' @param y A \code{numeric} or \code{factor} vector, indicating a continuous outcome or class label.
#' @param w A \code{numeric} vector, the sample weight. Will be 1 if left blank.
#'
#' @return An \code{dnnetInput} object.
#'
#' @importFrom methods new
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#' @export
importDnnet <- function(x, y, w = rep(1, length(y))) {

  new("dnnetInput", x = as.matrix(x), y = y, w = w)
}

#' Import Data to create a \code{dnnetSurvInput} object.
#'
#' @param x A \code{matrix} containing all samples/variables. It has to be \code{numeric}
#' and cannot be left blank. Any variable with missing value will be removed.
#' @param y A \code{numeric} vector, indicating the censoring time.
#' @param w A \code{numeric} vector, the sample weight. Will be 1 if left blank.
#' @param e A \code{numeric} vector, the event status.
#'
#' @return An \code{dnnetInput} object.
#'
#' @importFrom methods new
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#' @export
importDnnetSurv <- function(x, y, e, w = rep(1, length(y))) {

  new("dnnetSurvInput", x = as.matrix(x), y = y, w = w, e = e)
}

#' Import Data to create a \code{trtInput} object.
#'
#' @param x A \code{matrix} containing all samples/variables. It has to be \code{numeric}
#' and cannot be left blank. Any variable with missing value will be removed.
#' @param y A \code{numeric} vector, indicating the censoring time.
#' @param z The treatment assignment.
#'
#' @return An \code{trtInput} object.
#'
#' @importFrom methods new
#'
#' @seealso
#' \code{\link{trtInput-class}}
#' @export
importTrt <- function(x, y, z) {

  new("trtInput", x = as.matrix(x), y = y, z = z)
}

#' A function to generate indice
#'
#' @param split As in \code{\link{dnnetInput-class}}.
#' @param n Sample size
#'
#' @return Returns a integer vector of indice.
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#'
#' @export
getSplitDnnet <- function(split, n) {

  if(is.numeric(split) && length(split) == 1 && split < 1)
    split <- sample(n, floor(n * split))

  if(is.numeric(split) && length(split) == 1 && split > 1)
    split <- 1:split

  if(is.character(split) && length(split) == 1 && split == "bootstrap")
    split <- sample(n, replace = TRUE)

  split
}

#' A function to split the \code{dnnetInput} object into a list of two \code{dnnetInput} objects:
#' one names train and the other named valid.
#'
#' @param object A \code{dnnetInput} object.
#' @param split A character, numeric variable or a numeric vector declaring a way to split
#' the \code{dnnetInput}. If it's number between 0 and 1, all samples will be split into two subsets
#' randomly, with the \code{train} containing such proportion of all samples and \code{valid} containing
#' the rest. If split is a character and is "bootstrap", the \code{train} will be a bootstrap sample
#' of the original data set and the \code{valid} will contain out-of-bag samples. If split is a vector
#' of integers, the \code{train} will contain samples whose indice are in the vector, and \code{valid} will
#' contain the rest.
#'
#' @return Returns a list of two \code{dnnetInput} objects.
#'
#' @seealso
#' \code{\link{dnnetInput-class}}
#'
#' @export
splitDnnet <-function(object, split) {

  split <- getSplitDnnet(split, dim(object@x)[1])

  train <- object
  train@x <- object@x[split, ]
  train@y <- object@y[split]
  train@w <- object@w[split]
  if(class(object) == "dnnetSurvInput")
    train@e <- object@e[split]

  valid <- object
  valid@x <- object@x[-split, ]
  valid@y <- object@y[-split]
  valid@w <- object@w[-split]
  if(class(object) == "dnnetSurvInput")
    valid@e <- object@e[-split]

  list(train = train, valid = valid, split = split)
}

#' A function to split the \code{trtInput} object into a list of two \code{trtInput} objects:
#' one names train and the other named valid.
#'
#' @param object A \code{trtInput} object.
#' @param split A character, numeric variable or a numeric vector declaring a way to split
#' the \code{trtInput}. If it's number between 0 and 1, all samples will be split into two subsets
#' randomly, with the \code{train} containing such proportion of all samples and \code{valid} containing
#' the rest. If split is a character and is "bootstrap", the \code{train} will be a bootstrap sample
#' of the original data set and the \code{valid} will contain out-of-bag samples. If split is a vector
#' of integers, the \code{train} will contain samples whose indice are in the vector, and \code{valid} will
#' contain the rest.
#'
#' @return Returns a list of two \code{trtInput} objects.
#'
#' @seealso
#' \code{\link{trtInput-class}}
#'
#' @export
splitTrt <-function(object, split) {

  split <- getSplitDnnet(split, dim(object@x)[1])

  train <- object
  train@x <- object@x[split, ]
  train@y <- object@y[split]
  train@w <- object@z[split]

  valid <- object
  valid@x <- object@x[-split, ]
  valid@y <- object@y[-split]
  valid@w <- object@z[-split]

  list(train = train, valid = valid, split = split)
}
