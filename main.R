packrat::repos_create("./.CRAN")
packrat::repos_upload("./REMOTES", ".CRAN")
# packrat::repos_upload("./modules/core/data/wrangling", ".cran")
packrat::init(options = list(local.repos = "./.CRAN"))
# packrat::install_local("ransack.cli.ransack")
# install.packages("ransack.cli.ransack", type = "source")
devtools::install_local("./REMOTES", force = TRUE)
packrat::snapshot(ignore.stale = TRUE)
packrat::restore()
shell("set PIPENV_VENV_IN_PROJECT=1& set PIP_PROCESS_DEPENDENCY_LINKS=1& pipenv install --skip-lock")

# np <- import("numpy", convert = FALSE)
# a <- np$array(c(1:4))
# a %>%
#   {.$cumsum()} %>%
#   {.$cumsum()} %>%
#   py_to_r
# # %%
# print(hello)
