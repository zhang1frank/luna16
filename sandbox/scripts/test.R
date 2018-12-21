# %%
reticulate::use_virtualenv(getOption("venv"))
library(reticulate)
library(tidyverse)
py <- import("__builtin__")
# %%
Test <- py$type("Test", tuple(), dict(
  `__init__` = function(self, x) {
    py_set_attr(self, "a", x)
    return(NULL)
  }
))
test <- py_call(Test, 2)
test2 <- py_call(Test, 5)
# %%
# %%
Test
# %%
# list(
#   "Test", tuple(), dict(
#     `__init__` = function(self, x) {
#       py_set_attr(self, "a", x)
#       return(NULL)
#     }
#   )
# ) %>%
#   do.call(py$type, .) %>%
#   py_call(7) %>%
#   .$a
# print(py_call(dir))
Test2 <- py$type("Test2", tuple(Test), dict(
  "__init__" = function(self, x) {
    Test$`__init__`(self, x)
    return(NULL)
  },
  "b" = function(self) {
    return(self$a)
  }
))

test3 <- py_call(Test2, 6)
test3$b()
# %%
data <- data %>% (function(.)
    .$as_in_context(model_ctx)) %>% (function(.)
    .$reshape(c(py$int(-1), py$int(784))))
