script_dir_path <- dirname(
  strsplit(commandArgs(trailingOnly = FALSE)[4], "=")[[1]][2]
)

root_dir <- getOption("root_dir")

## A packrat directory has to exist in the calling directory or later will fail
packrat::packrat_mode(on = TRUE, project = root_dir)
## Have to turn off or later packrat calls will fail
packrat::packrat_mode(on = FALSE)
## Source from app's private packrat
packrat::packrat_mode(on = TRUE, project = script_dir_path)
## Need to run restore to hydrate the cli tool after install
if (any(is.na(packrat::status(quiet = TRUE)[, "library.version"]))) {
  packrat::restore()
}

## no package references until below this line
##############################################
magics::.__file__(function(x) {
  x
})

import::here("%>%", "%T>%",
  .from = "magrittr")

modules::expose(argv,
  module = file.path(.__file__, "R.modules/client.R"))
modules::expose(hoist,
  module = file.path(.__file__, "R.modules/hoist.R"))
modules::expose(clean,
  module = file.path(.__file__, "R.modules/clean.R"))
modules::expose(cranify,
  module = file.path(.__file__, "R.modules/cranify.R"))
modules::expose(packify,
  module = file.path(.__file__, "R.modules/packify.R"))

switch(argv$`<command>`,
  "hoist" = hoist(),
  "clean" = clean(),
  "cranify" = cranify(),
  "packify" = {
    # ensure that all commands are applied to project packrat and not ransack's
    packrat::packrat_mode(on = TRUE, project = root_dir)
    lapply(argv$e, function(el) {
      c(list(), "path" = if (is.na(el)) "." else el) %>%
      packify()
    })
  },
  "snapshot" = {
    packrat::packrat_mode(on = TRUE, project = root_dir)
    packrat::snapshot(ignore.stale = TRUE)
  }
)
