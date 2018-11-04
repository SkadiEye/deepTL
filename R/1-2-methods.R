###########################################################
### Define generic functions: show, print, '[', '$', plot

#################################
#### dnnetInput class

#' @describeIn dnnetInput Method to show \code{dnnetInput} object.
#'
#' @param object A \code{dnnetInput} object.
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

#' @describeIn dnnet Method to show \code{dnnet} object.
#'
#' @param object A \code{dnnet} object.
#'
#' @export
setMethod("show",
          "dnnet",
          function(object) {

            print("A dnnet model object. ")

          })
