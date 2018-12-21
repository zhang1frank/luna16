magics::.__file__(function(x) {
  x
})

modules::export("clean")

scrub <- function() {
  root_dir <- getOption("root_dir")
  while (dir.exists(file.path(root_dir, ".cran"))) {
    R.utils::removeDirectory(
      file.path(root_dir, ".cran"),
      recursive = TRUE, mustExist = FALSE
    )
  }
  tryCatch({
      packrat::packrat_mode(on = TRUE, project = root_dir)
      packrat::disable(root_dir)
    },
    error = function(e) {
      ## disable needs an inited packrat, so we make a dummy packrat
      packrat::init(project = root_dir, infer.dependencies = FALSE)
      packrat::disable(root_dir)
    }
  )
  R.utils::removeDirectory(
    file.path(root_dir, "packrat"),
    recursive = TRUE, mustExist = FALSE
  )
  ## need to use remove_directory to eliminate symlinks at the base
  if (dir.exists(file.path(root_dir, "packrat"))) {
    lapply(list.dirs(file.path(root_dir, "packrat"), recursive = TRUE),
      function(el) {
        # some nonsense where files were not getting deleted, had to use rm
        if (!any(dir.exists(file.path(el, list.files(el))))) {
          lapply(file.path(el, list.files(el)), function(x) {
            system(sprintf("sh -c 'rm %s'", x))
          })
        }
        R.utils::removeDirectory(
          el, recursive = TRUE, mustExist = FALSE
        )
      }
    )
  }
  while (dir.exists(file.path(root_dir, "packrat"))) {
    R.utils::removeDirectory(
      file.path(root_dir, "packrat"),
      recursive = TRUE, mustExist = FALSE
    )
  }
}

clean <- function() {
  scrub()
  # tryCatch({
  #   R.utils::withTimout(scrub(), timeout = 36000000, onTimeout = "error")
  # }
  # , error = function(e) {
  #   print("Timeout after 60 sec! Rescrubbing...")
  #   scrub()
  # })
}
