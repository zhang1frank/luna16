# download from cran
local({
  r <- getOption("repos")
  r["CRAN"] <- "http://cran.r-project.org"
  options(repos = r)
})
# preconfigured options
options(
  # allows getSrcDirectory to work
  keep.source = TRUE,
  # suppress warnings
  warn = -1
)
# custom options
options(
  # provide root path for ransack
  root_dir = getwd()
)
#### -- Packrat Autoloader (version 0.4.9-2) -- ####
source("packrat/init.R")
#### -- End Packrat Autoloader -- ####
