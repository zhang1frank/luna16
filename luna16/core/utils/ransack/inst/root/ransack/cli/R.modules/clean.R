import::here(remove_directory = "removeDirectory", .from = "R.utils")

modules::export("clean")

clean <- function() {
  remove_directory("./.cran", recursive = TRUE, mustExist = FALSE)
  packrat::packrat_mode(on = TRUE, project = ".")
  packrat::disable(".")
  remove_directory("./packrat", recursive = TRUE, mustExist = FALSE)
  ## need to use remove_directory to eliminate symlinks at the base
  lapply(list.dirs("./packrat", recursive = TRUE), function(el) {
    remove_directory(
      el, recursive = TRUE, mustExist = FALSE
    )
  })
  remove_directory("./packrat", recursive = TRUE, mustExist = FALSE)
}
