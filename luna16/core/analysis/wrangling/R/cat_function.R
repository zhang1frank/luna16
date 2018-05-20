#' A cat function
#' @export
cat_function <- function(love = TRUE) {
  # if (love == TRUE) print("I love cats") else print ("I am not a cool person")
  # print(packageName())
  system.file("root", "test.txt", package = packageName())
}
