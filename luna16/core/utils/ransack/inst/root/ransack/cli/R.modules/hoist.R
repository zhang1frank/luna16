## Needed for document
modules::import(roxygen2)
modules::import(desc)
## Needed for install_local
modules::import(git2r)

import::here(list.parse, .from = "rlist")
import::here(read_file, .from = "readr")
import::here("%>%", "%T>%", .from = "magrittr")
import::here(install_local, document, .from = "devtools")
import::here(create_link = "createLink", .from = "R.utils")

modules::export("hoist")

hoist <- function() {
  packrat::repos_create("./.cran") %>% {
    read_file("./RANSACK")
  } %>%
  list.parse(type = "yaml") %>%
  lapply(
    function(el) {
      package_path <- packrat::repos_upload(el$path, ".cran")
      c(el, "package" = sub("_.*", "", basename(package_path)))
    }
  ) %T>% {
    packrat::init(options = list(local.repos = "./.cran"))
  } %T>%
  lapply(
    function(el) {
      if (dir.exists(file.path(packrat::lib_dir(), el$package))) {
        document(el$path)
        install_local(el$path, force = TRUE)
        if (isTRUE(el$editable)) {
          create_link(
            link = file.path(
              packrat::lib_dir(), el$package, "root"
            ),
            target = file.path(el$path, "inst/root"),
            overwrite = TRUE
          )
        }
      }
    }
  )
}
