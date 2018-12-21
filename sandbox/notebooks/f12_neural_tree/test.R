# %%
magics::.__file__(function(x) {
  x
})

reticulate::use_virtualenv(getOption("venv"))

mx <- reticulate::import("mxnet", convert = FALSE)
np <- reticulate::import("numpy", convert = FALSE)
nd <- mx$nd

py <- reticulate::import("builtins")

modules::expose(TreeNode,
  module = file.path(.__file__, "R.modules/TreeNode.R"))

modules::expose(NeuralTree,
  module = file.path(.__file__, "R.modules/NeuralTree.R"))

# %%
(function(x = nd$array(list(
  c(1, 2, 3, 4, 5),
  c(3, 4, 3, 4, 1),
  c(0, 0, 1, 4, 0),
  c(3, 3, 7, 5, 9),
  c(3, 2, 8, 4, 1),
  c(3, 1, 6, 7, 8)
)),
y = nd$array(list(
  c(0, 0, 0, 1, 0)
))) {
  test <- TreeNode(tau = nd$array(list(5)))
  test$`_tau`$data()
  test$`_update_extent`(x)
  print(test$`_min_list`$data())
  print(test$`_max_list`$data())
  print(test$`_tau`$data())
  test$`_update_extent`(y)
  print(test$`_min_list`$data())
  print(test$`_max_list`$data())
  test$`_init_param`(test$`_split`, nd$array(list(3.5)))
  print(test$`_split`$data())
  print(test$`_sharpness`$data())
  test$`_create_split`(nd$array(list(2)), (nd$array(list(3.5))))
  print(test$`_split`$data())
  print(test$forward(x))
})()

# %%
tree <- NeuralTree()
reticulate::py_len(tree$collect_params()$`_params`)

# %%
library(magrittr)
reticulate::tuple(NULL, NULL)[0]

# %%

(function () {
  return(1)
}) %>% (function (x) {
  x()
})()

reticulate::tuple(1) %>% is.null()

# %%
tryCatch({
  py$type("_",
    reticulate::tuple(),
    reticulate::dict(
      "__init__" = function(self, x) {
        reticulate::py_set_attr(self, "_", x)
        return(NULL)
      }
    )
  )(NULL)
}
, error = (function() function(e) a <- 1)())

ModClass <- py$type("ModClass",
  reticulate::tuple(),
  reticulate::dict(
    static_method = (function () {
      print("I am static")
    }) %>% py$staticmethod(),
    class_method = (function (cls) {
      cls$static_method()
    }) %>% py$classmethod(),
    instance_method = function(self) {
      self$class_method()
    }
  )
)

# %%
ModClass$static_method()
ModClass$class_method()
ModClass()$instance_method()

# %%
nd$broadcast_maximum(nd$array(list(0, 5, 5, -1)), nd$array(list(1)))