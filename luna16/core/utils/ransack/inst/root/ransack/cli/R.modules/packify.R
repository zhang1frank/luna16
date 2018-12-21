magics::.__file__(function(x) {
  x
})

modules::export("packify")

packify <- function(el) {
  if (!isTRUE(
    el$package == (
      package <- desc::desc(el$path)$get("Package")
    ))
  ) {
    el$package <- package[["Package"]]
  }
  devtools::install_local(el$path, force = TRUE)
  devtools::document(el$path)
  devtools::install_local(el$path, force = TRUE)
  if (dir.exists(file.path(el$path, "inst/root")) &
  dir.exists(packrat::lib_dir())) {
    R.utils::createLink(
      link = file.path(
        packrat::lib_dir(), el$package, "root"
      ),
      target = file.path(el$path, "inst/root"),
      overwrite = TRUE
    )
  }
  if (dir.exists(file.path(el$path, "inst/dist")) &
    dir.exists(
      file.path(packrat::project_dir(), ".dist", basename(el$path))
    )
  ) {
    R.utils::createLink(
      link = file.path(
        packrat::lib_dir(), el$package, "dist"
      ),
      target = file.path(
        packrat::project_dir(), ".dist", basename(el$path)
      ),
      overwrite = TRUE
    )
  }
  return(el)
}
