## usethis namespace: start
#' @useDynLib mcmcpackage, .registration = TRUE
## usethis namespace: end
NULL

#' this does something
#' @export
hello <- function() {
  print("Hello, world!")
}
