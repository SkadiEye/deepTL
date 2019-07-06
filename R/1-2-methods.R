###########################################################
### Define generic functions: show, print, '[', '$', plot

#################################
#### dnnetInput class

#' @name show
#' @rdname show
#'
#' @title Method show for the package
#'
#' @description The method show for \code{dnnetInput} or \code{dnnet} object.
#'
#' @param object A \code{dnnet} or \code{dnnetInput} object.
#'
#' @seealso
#' \code{\link{dnnetInput-class}}\cr
#' \code{\link{dnnet-class}}\cr
#'
NULL

#' @rdname show
#' @importFrom methods show
#'
#' @export
setMethod("show",
          "dnnetInput",
          function(object) {

            print("y: (first 6)")
            print(object@y[1:6])
            print("x: (first 6x6)")
            print(paste("dim(x) =", dim(x)[1], "x", dim(x)[2]))
            print(object@x[1:6, 1:6])
            print("w: (first 6)")
            print(object@w[1:6])
          })

#################################
#### dnnet class

#' @rdname show
#' @importFrom methods show
#'
#' @export
setMethod("show",
          "dnnet",
          function(object) {

            print("A dnnet model object. ")

          })
