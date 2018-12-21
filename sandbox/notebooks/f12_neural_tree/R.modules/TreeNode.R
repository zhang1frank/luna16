# %%
mx <- reticulate::import("mxnet", convert = FALSE)
nd <- mx$nd
Block <- reticulate::import("mxnet", convert = FALSE)$gluon$Block

py <- reticulate::import("builtins")

modules::import(magrittr)

# %%
TreeNode <- py$type("TreeNode",
    reticulate::tuple(Block),
    reticulate::dict(
        "__init__" = function(self,
            split = NULL,
            min_list = NULL, max_list = NULL, tau = NULL,
            parent = NULL, left = NULL, right = NULL,
            ...
        ) {
            Block$`__init__`(self, ...)
            reticulate::py_set_attr(self, "_parent", parent)
            reticulate::py_set_attr(self, "_left", left)
            reticulate::py_set_attr(self, "_right", right)
            with(self$name_scope(), {
              reticulate::py_set_attr(self, "_sharpness",
                self$params$get("sharpness") %T>%
                  self$`_init_param`(nd$array(list(1)))
              )
              reticulate::py_set_attr(self, "_split",
                self$params$get("split") %T>% self$`_init_param`(split)
              )
              reticulate::py_set_attr(self, "_min_list",
                self$params$get("min_list", grad_req = "null") %T>%
                  self$`_init_param`(min_list)
              )
              reticulate::py_set_attr(self, "_max_list",
                self$params$get("max_list", grad_req = "null") %T>%
                  self$`_init_param`(max_list)
              )
              reticulate::py_set_attr(self, "_tau",
                self$params$get("tau", grad_req = "null") %T>%
                  self$`_init_param`(tau)
              )
            })
            return(NULL)
        },
        "_init_param" = function(self, param, value) {
          if (!is.null(value %>% reticulate::py_to_r())) {
            param$set_data(value)
            param$initialize(init = mx$init$Constant(value))
          }
        },
        "_update_extent" = function(self, x) {
          nd$min(x, axis = 0 %>% as.integer) %>% {
            if (self$`_min_list`$shape
              %>% reticulate::py_to_r() %>% is.null()) {
              self$`_init_param`(self$`_min_list`, .)
            } else {
              nd$minimum(., self$`_min_list`$data()) %>%
                self$`_min_list`$set_data()
            }
          }
          nd$max(x, axis = 0 %>% as.integer) %>% {
            if (self$`_max_list`$shape %>%
              reticulate::py_to_r() %>% is.null()) {
              self$`_init_param`(self$`_max_list`, .)
            } else {
              nd$maximum(., self$`_max_list`$data()) %>%
                self$`_max_list`$set_data()
            }
          }
        },
        "_update_tau" = function(self) {
          nd$add(self$`_max_list`, self$`_min_list`) %>%
            nd$sum() %>% nd$reciprocal() %>%
            nd$random$exponential() %>% {
              if (self$`_parent` %>% is.null()) {
                return(.)
              } else {
                nd$add(., self$`_parent`$`_tau`$data())
              }
            } %>% {
              if (self$`_tau`$shape %>%
                reticulate::py_to_r() %>% is.null()) {
                self$`_init_param`(self$`_tau`, .)
              } else {
                self$`_tau`$set_data(.)
              }
            }
        },
        "__split_data" = (
          function(cls, x, dim, split, y = NULL) {
            split <- nd$pick(x,
              dim %>% nd$broadcast_to(x$shape[0]),
              keepdims = TRUE
            ) %>%
              nd$subtract(split) %>%
              nd$tanh()
            cls$`__shard`(split, x, y)
          }
        ) %>% py$classmethod(),
        "_split_node" = function(self, x, y = NULL) {
          # determine activation of node based on its split value
          split <- self$forward(x)
          self$`__shard`(split, x, y)
        },
        "__shard" = (
          function(split, x, y = NULL) {
            # sort the data based on the activation values
            splitsortorder <- split %>% nd$argsort()
            reorderedx <- x$`__getitem__`(splitsortorder)
            if (!(y %>% is.null())) {
              reorderedy <- y$`__getitem__`(splitsortorder)
            }
            reorderedsplit <- split$`__getitem__`(splitsortorder)
            # if all of data is greater than threshold, send all data to right
            if (reorderedsplit$`__getitem__`(0 %>% as.integer)$asnumpy() %>%
              reticulate::py_to_r() > 0
            ) {
              if (y %>% is.null()) {
                list(NULL, reorderedx)
              } else {
                list(NULL, list(reorderedx, reorderedy))
              }
            # if all of data is less than threshold, send all data to left
            } else if (
              reorderedsplit$`__getitem__`(-1 %>% as.integer)$asnumpy() %>%
                reticulate::py_to_r() < 0
            ) {
              if (y %>% is.null()) {
                list(reorderedx, NULL)
              } else {
                list(list(reorderedx, reorderedy), NULL)
              }
            # determine the highest index belonging to a negative
            # number based on multiplying argsort of the pre sorted split list
            # w split list sign, that index + 1 is used to slice the data into 2
            } else {
              splitpt <- (
                nd$multiply(
                  nd$argsort(reorderedsplit), nd$sign(reorderedsplit)
                ) %>%
                  nd$argsort()
              )[0] %>% nd$add(nd$array(list(1)))
              leftx <- nd$slice_axis(
                reorderedx,
                axis = 0 %>% as.integer,
                begin = 0 %>% as.integer,
                end = splitpt$asnumpy() %>% reticulate::py_to_r() %>%
                  as.integer
              )
              rightx <- nd$slice_axis(
                reorderedx,
                axis = 0 %>% as.integer,
                begin = splitpt$asnumpy() %>% reticulate::py_to_r() %>%
                  as.integer,
                end = NULL
              )
              lefty <- nd$slice_axis(
                reorderedy,
                axis = 0 %>% as.integer,
                begin = 0 %>% as.integer,
                end = splitpt$asnumpy() %>% reticulate::py_to_r() %>%
                  as.integer
              )
              righty <- nd$slice_axis(
                reorderedy,
                axis = 0 %>% as.integer,
                begin = 0 %>% as.integer,
                end = splitpt$asnumpy() %>% reticulate::py_to_r() %>%
                  as.integer
              )
              if (y %>% is.null()) {
                list(leftx, rightx)
              } else {
                list(list(leftx, lefty), list(rightx, righty))
              }
            }
          }
        ) %>% py$staticmethod(),
        "_create_split" = function(self, dim, split) {
          self$`_init_param`(self$`_split`, split)
          reticulate::py_set_attr(self, "forward",
            function(x) {
              nd$pick(x,
                dim %>% nd$broadcast_to(x$shape[0]),
                keepdims = TRUE
              ) %>%
                nd$subtract(self$`_split`$data()) %>%
                nd$multiply(self$`_sharpness`$data() %>% nd$relu()) %>%
                nd$tanh()
            }
          )
        },
        "forward" = function(self, x) {
        }
    )
)

# %%
modules::export("TreeNode")