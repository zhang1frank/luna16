magics::.__file__(function(x) {
  x
})

modules::expose(cranify,
  module = file.path(.__file__, "cranify.R"))
modules::expose(packify,
  module = file.path(.__file__, "packify.R"))

import::here("%>%", "%T>%",
  .from = "magrittr")

modules::export("hoist")

hoist <- function() {
  root_dir <- getOption("root_dir")
  cranify() %T>% {
    packrat::init(project = root_dir, options = list(local.repos = "./.cran"))
  } %T>%
  lapply(packify(el)) %T>% {
    packrat::snapshot(ignore.stale = TRUE)
  }
}
