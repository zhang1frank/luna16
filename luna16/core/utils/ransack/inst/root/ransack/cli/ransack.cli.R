script_dir_path <- dirname(
  strsplit(commandArgs(trailingOnly = FALSE)[4], "=")[[1]][2]
)

## A packrat directory has to exist in the calling directory or later will fail
packrat::packrat_mode(on = TRUE, project = ".")
## Have to turn off or later packrat calls will fail
packrat::packrat_mode(on = FALSE)
## Source from app's private packrat
packrat::packrat_mode(on = TRUE, project = script_dir_path)
## Need to run restore to hydrate the cli tool after install
if (any(is.na(packrat::status(quiet = TRUE)[, "library.version"]))) {
  packrat::restore()
}

modules::expose(argv, module = file.path(script_dir_path, "R.modules/client.R"))
modules::expose(hoist,
  module = file.path(script_dir_path, "R.modules/hoist.R"))
modules::expose(clean,
  module = file.path(script_dir_path, "R.modules/clean.R"))

switch(argv$`<command>`,
  "hoist" = hoist(),
  "clean" = clean()
)
