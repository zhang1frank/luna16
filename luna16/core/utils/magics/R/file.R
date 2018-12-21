#' @export
.__file__ <- function(x, noisy = NULL) {
  assign(".__file__",
    env = parent.frame(),
    value = (function() {
      if (is.null(noisy)) print <- function(x) {x}
      if (rapportools::is.empty(src <- utils::getSrcDirectory(x)))
        if (rapportools::is.empty(scr <- funr::get_script_path()))
          print(getwd())
        else print(scr)
      else print(src)
    })()
  )
}
## devtools::document("luna16/core/utils/magics")
# devtools::install_local("luna16/core/utils/magics", force=TRUE)
