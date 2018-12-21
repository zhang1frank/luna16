magics::.__file__(function(x) {
  x
})

import::here("%>%",
  .from = "magrittr")

modules::export("cranify")

cranify <- function() {
  root_dir <- getOption("root_dir")
  while (dir.exists(file.path(root_dir, ".cran"))) {
    R.utils::removeDirectory(
      file.path(root_dir, ".cran"),
      recursive = TRUE, mustExist = FALSE
    )
  }
  packrat::repos_create(
    file.path(root_dir, ".cran")
  ) %>% {
    readr::read_file(file.path(root_dir, "RANSACK"))
  } %>%
    rlist::list.parse(type = "yaml") %>%
    lapply(
      function(el) {
        package_path <- packrat::repos_upload(
          file.path(root_dir, el$path), ".cran"
        )
        c(el, "package" = sub("_.*", "", basename(package_path)))
      }
    )
}
