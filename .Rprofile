# download from cran
local({
  r <- getOption("repos")
  r["CRAN"] <- "http://cran.r-project.org"
  options(repos = r)
})
# provide proper path for reticulate
options(venv = system("pipenv --venv", intern = TRUE))
# suppress warnings
options(warn = -1)
#### -- Packrat Autoloader (version 0.4.9-2) -- ####
source("packrat/init.R")
#### -- End Packrat Autoloader -- ####
