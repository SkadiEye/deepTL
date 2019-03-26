#################################
#### Activation function ####

#' @name actF
#' @rdname actF
#'
#' @title Activation functions and their first order derivatives
#'
#' @description A collection of activation functions and their first
#'  order derivatives used in deep neural networks
#'
#' @details
#'
#' Sigmoid Function:
#' sigmoid(x) = 1/(1+exp(-x))
#'
#' Hyperbolic Tangent Function:
#' tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
#'
#' Rectified Linear Units:
#' relu(x) = max(0, x)
#'
#' Leaky ReLU:
#' prelu(x, a) = max(x*a, x), (a<1)
#'
#' Exponential Linear Units:
#' elu(x, alpha) = max(alpha*(exp(x)-1), x), (alpha<=1)
#'
#' Continuously Differentiable Exponential Linear Units
#' celu(x, alpha) = max(alpha*(exp(x/alpha)-1), x)
#'
#' @param x Input of the activation function
#' @param a a or alpha in the function
#'
#' @return Returns the value after activation
#'
#' @seealso
#' \code{\link{dnnet}}\cr
NULL

#' @rdname actF
#' @section Methods (by generic):
#' Sigmoid function.
#' @export
sigmoid <- function(x) {1/(exp(-x)+1)}

#' @rdname actF
#' @section Methods (by generic):
#' First order derivative of Sigmoid function.
#' @export
sigmoid_ <- function(x) {y <- sigmoid(x); y-y**2}

#' @rdname actF
#' @section Methods (by generic):
#' Tanh function.
#' @export
tanh <- function(x) {base::tanh(x)}

#' @rdname actF
#' @section Methods (by generic):
#' First order derivative of tanh function.
#' @export
tanh_ <- function(x) {y <- tanh(x); 1-y**2}

#' @rdname actF
#' @section Methods (by generic):
#' ReLU.
#' @export
relu <- function(x) {(abs(x) + x)/2}

#' @rdname actF
#' @section Methods (by generic):
#' First order derivative of ReLU.
#' @export
relu_ <- function(x) {(x > 0)*1}

#' @rdname actF
#' @section Methods (by generic):
#' Leaky ReLU.
#' @export
prelu <- function(x, a = 0.2) {
  m <- (1+a)/2
  n <- (1-a)/2
  (abs(x)*m) + (x*n)
}

#' @rdname actF
#' @section Methods (by generic):
#' First order derivative of leaky ReLU.
#' @export
prelu_ <- function(x, a = 0.2) {
  b <- 1-a
  return((x > 0)*b + a)
}

#' @rdname actF
#' @param alpha A pre-specified numeric value less or equal to 1.
#' @section Methods (by generic):
#' ELU.
#' @export
elu <- function(x, a = 1) {(x > 0)*x + a*(exp((x <= 0)*x) - 1)}

#' @rdname actF
#' @section Methods (by generic):
#' First order derivative of ELU function.
#' @export
elu_ <- function(x, a = 1) {(x > 0) + a*(exp((x <= 0)*x) - (x <= 0))}

#' @rdname actF
#' @param alpha A pre-specified numeric value less or equal to 1.
#' @section Methods (by generic):
#' CELU.
#' @export
celu <- function(x, a = 1) {(x > 0)*x + a*(exp((x <= 0)*x/a) - 1)}

#' @rdname actF
#' @section Methods (by generic):
#' First order derivative of CELU function.
#' @export
celu_ <- function(x, a = 1) {(x > 0) + exp((x <= 0)*x/a) - (x <= 0)}
