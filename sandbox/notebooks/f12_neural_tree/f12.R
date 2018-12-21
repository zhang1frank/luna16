# %%
reticulate::use_virtualenv(getOption("venv"))

mx <- reticulate::import("mxnet", convert = FALSE)
np <- reticulate::import("numpy", convert = FALSE)
nd <- mx$nd
autograd <- mx$autograd
gluon <- mx$gluon
nn <- mx$gluon$nn
Block <- mx$gluon$Block

py <- reticulate::import("builtins")
modules::import(foreach)
modules::import(magrittr)
modules::import(gsubfn)

ctx <- mx$cpu()

tryCatch({
    py$type("_", reticulate::tuple(), reticulate::dict(
      "__init__" = function(self, x) {
        reticulate::py_set_attr(self, "_", x)
        return(NULL)
      }
  ))(NULL)
}, error = function (e) {i<-1})
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
            if (self$`_min_list`$shape %>% reticulate::py_to_r() %>% is.null()) {
              self$`_init_param`(self$`_min_list`, .)
            } else {
              nd$minimum(., self$`_min_list`$data()) %>%
                self$`_min_list`$set_data()
            }
          }
          nd$max(x, axis = 0 %>% as.integer) %>% {
            if (self$`_max_list`$shape %>% reticulate::py_to_r() %>% is.null()) {
              self$`_init_param`(self$`_max_list`, .)
            } else {
              nd$maximum(., self$`_max_list`$data()) %>%
                self$`_max_list`$set_data()
            }
          }
        },
        "_create_split" = function(self, dim, split) {
          self$`_init_param`(self$`_split`, split)
          reticulate::py_set_attr(self, "forward",
            function(x) {
              nd$pick(x, dim %>% nd$broadcast_to(x$shape[0]), keepdims = TRUE) %>%
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

# %%
(function(x = nd$array(list(
  c(1,2,3,4,5),
  c(3,4,3,4,1),
  c(0,0,1,4,0),
  c(3,3,7,5,9),
  c(3,2,8,4,1),
  c(3,1,6,7,8)
)),
y = nd$array(list(
  c(0,0,0,1,0)
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
(function(x = nd$array(list(
  c(1,2,3,4,5),
  c(3,4,3,4,1),
  c(0,0,1,4,0),
  c(3,3,7,5,9),
  c(3,2,8,4,1),
  c(3,1,6,7,8)
))) {
  y <- nd$pick(x, nd$array(list(3)) %>% nd$broadcast_to(x$shape[0]))
  print(split <- nd$tanh(nd$subtract(y, 3.5)))
  print(splitsortorder <- split %>% nd$argsort())
  print(reorderedx <- x$`__getitem__`(splitsortorder))
  print(reorderedsplit <- split$`__getitem__`(splitsortorder))
  if (reorderedsplit$`__getitem__`(0 %>% as.integer)$asnumpy() %>% reticulate::py_to_r() > 0) {
    print(right <- reorderedx)
    print("right")
  } else if (reorderedsplit$`__getitem__`(-1 %>% as.integer)$asnumpy() %>% reticulate::py_to_r() < 0) {
    print(left <- reorderedx)
    print("left")
  } else {
    print(splitpt <- (nd$multiply(nd$argsort(reorderedsplit), nd$sign(reorderedsplit)) %>% nd$argsort())[0] %>% nd$add(nd$array(list(1))))
    print(left <- nd$slice_axis(reorderedx, axis = 0 %>% as.integer, begin = 0 %>% as.integer, end = splitpt$asnumpy() %>% reticulate::py_to_r() %>% as.integer))
    print(right <- nd$slice_axis(reorderedx, axis = 0 %>% as.integer, begin = splitpt$asnumpy() %>% reticulate::py_to_r() %>% as.integer, end = NULL))
  }
})()
# %%
NeuralTree <- py$type("NeuralTree",
    reticulate::tuple(Block),
    reticulate::dict(
      "__init__" = function(self,
          stop_split = function(x) FALSE,
          pick_best_split = function(x) FALSE,
          draw_split = function(x) FALSE,
          draw_feature = function(x) FALSE,
          draw_data = function(x) FALSE,
          ...
      ) {
          Block$`__init__`(self, ...)
          # utility functions
          reticulate::py_set_attr(self, "_stop_split", stop_split)
          reticulate::py_set_attr(self, "_pick_best_split", pick_best_split)
          reticulate::py_set_attr(self, "_draw_split", draw_split)
          reticulate::py_set_attr(self, "_draw_feature", draw_feature)
          reticulate::py_set_attr(self, "_draw_data", draw_data)
          # choose when to grow the tree
          reticulate::py_set_attr(self, "_is_growing", FALSE)
          # create root node
          # reticulate::py_set_attr(self, "_root_node", TreeNode())
          # cannot create root node in init, otherwise initialize will complain
          with(self$name_scope(), {
              # decision layer
              reticulate::py_set_attr(self, "_decision_layer",
                  gluon$contrib$nn$Concurrent()
              )
              # leaf layer
              reticulate::py_set_attr(self, "_leaf_layer",
                  gluon$contrib$nn$Concurrent()
              )
          })
          return(NULL)
      },
      "_recurse_node" = function(self, x, node) {
        # determine activation of node based on its split value
        split <- node(x)
        # sort the data based on the activation values
        splitsortorder <- split %>% nd$argsort()
        reorderedx <- x$`__getitem__`(splitsortorder)
        reorderedsplit <- split$`__getitem__`(splitsortorder)
        # if all of data is greater than threshold, send all data to right
        if (reorderedsplit$`__getitem__`(0 %>% as.integer)$asnumpy() %>%
          reticulate::py_to_r() > 0
        ) {
          self$`_grow_node`(reorderedx, node$`_right`)
        # if all of data is less than threshold, send all data to left
        } else if (reorderedsplit$`__getitem__`(-1 %>% as.integer)$asnumpy() %>%
          reticulate::py_to_r() < 0
        ) {
          self$`_grow_node`(reorderedx, node$`_left`)
        # determine the highest index belonging to a negative
        # number based on multiplying the argsort of the pre sorted split list
        # with split list sign, that index + 1 is used to slice the data into 2
        } else {
          splitpt <- (
            nd$multiply(nd$argsort(reorderedsplit), nd$sign(reorderedsplit)) %>%
              nd$argsort()
          )[0] %>% nd$add(nd$array(list(1)))
          nd$slice_axis(
            reorderedx,
            axis = 0 %>% as.integer,
            begin = 0 %>% as.integer,
            end = splitpt$asnumpy() %>% reticulate::py_to_r() %>% as.integer
          ) %>%
            self$`grow_node`(node$`_left`)
          nd$slice_axis(
            reorderedx,
            axis = 0 %>% as.integer,
            begin = splitpt$asnumpy() %>% reticulate::py_to_r() %>% as.integer,
            end = NULL
          ) %>%
            self$`grow_node`(node$`_right`)
        }
      },
      "_grow_node" = function(self, x, node) {
        if (!(function () {
          # check if node has a split time, if not, don't bother sampling
          if (node$`_tau`$shape %>% reticulate::py_to_r() %>% is.null() ||
            # check if additional data extent == 0, if so, don't bother sampling
            nd$add(
              nd$subtract(node$`_min_list`$data(), x) %>%
                nd$maximum(nd$array(list(0))),
              nd$subtract(x, node$`_max_list`$data()) %>%
                nd$maximum(nd$array(list(0)))
            ) %>%
              { .$asnumpy() } == 0
          # invoke extend node with current node
          ) FALSE else self$`_extend_ node`(x, node)
          # check to make sure if node was actually extended
        })()) {
          # if extend node not extended
          # update the node boundaries with new data
          node$`_update_extent`(x)
          # if no split time, invoke split node
          if (node$`_tau`$shape %>% is.null()) {
            self$`_split_node`(x, node)
          # if there is split time, recurse on children
          } else {
            self$`_recurse_node`(x, node)
          }
        }
      },
      "_split_node" = function(self, x, node) {
          # check to split or not
          # if don't split, add node to leaf layer, end
          # if split
            # determine split time based on node boundaries
            # sample feature
            # sample split
            # determine best split
            # set split on node, change forward based on feature, add node to decision layer
            # create children nodes, blanks
            # shard the data based on the split
            # invoke grow node with left node
            # invoke grow node with right node

          # with(self$name_scope(), {
          #     if (self$`_stop_split`() == TRUE) {
          #         self$leaf_layer$add(node)
          #     } else {
          #         # obtain split and split dim
          #         list[split, axis] <- x %>%
          #           self$draw_feature() %>%
          #           self$draw_split() %>%
          #           self$pick_best_split()
          #         # assign the split
          #         node$params$get("split")$set_data(split)
          #         # add node to network
          #         self$decision_layer$add(node)
          #         # recurse on left
          #         l_x <-
          #         l_node <- self$`_spawn_node`(l_x, parent = node)
          #         reticulate::py_set_attr(self, "left", l_node)
          #         self$`_split_node`(l_x, l_node)
          #         # recurse on right
          #         r_x <-
          #         r_node <- self$`_spawn_node`(r_x, parent = node)
          #         reticulate::py_set_attr(self, "right", r_node)
          #         self$`_split_node`(r_x, r_node)
          #     }
          # })
      },
      "_extend_node" = function(self, x, node) {
          # return true if node is extended
          # determine additional data extent compared to node
          # if node has no parent, then time split of parent = 0
          # sample time based on additional data extent
          # if (split of parent + sampled time < current node split time)
            # sample features
            # sample split
            # choose best split
            # create a new parent node with min, max,
              # previous parent must now have new parent as child
              # new parent has previous parent as parent
              # new parent has node as the appropriate child based on axis and whether bounds are both greater than or less than parent split
              # add new parent to decision layer
              # create missing sibling node, blank
              # invoke grow node with both sibling nodes
      },
      "forward" = function(self, x, ...) {
        # if the tree is empty, initialize first node
        if (reticulate::py_len(self$collect_params()$`_params`) == 0) {
          reticulate::py_set_attr(self, "_root_node", TreeNode())
        }
        # check if want to grow out tree
        if (self$`_is_growing` == TRUE) {
          # if you want to bootstrap, or do some other form of sample selection for training
          self$`_grow_node`(self$`_draw_data`(x, ...), self$`_root_node`)
        }
        x %>% self$decision_layer() %>% self$leaf_layer()
      }
    ))
# %%
test <- TreeNode()
nd$minimum(test$params$get("min_list")$data(), nd$array(list(1)))
test$params$get("min_list") %>% {.$set_data(nd$array(list(1)))}
nd$minimum(nd$array(list(1,2,3,-1)), nd$array(list(0))) %>% nd$min()
# %%
test$params$get("min_list") %>% {
  .$initialize(init = mx$init$Constant(nd$array(list(0.5))))
}
# %%
test$params$get("min_list")$data()
(x <- nd$array(list(c(1,2,-1), c(7,6,1), c(-2, 10,0))))
nd$argsort(x, axis = 1 %>% as.integer)
test$`_update_extent`(x)
test$`_min_list`$data()
test$`_max_list`$data()
# %%
1 %>% c(5, c(3,.,2))
nd$array(list(c(1,2,3), c(4,5,6)))$`__getitem__`(1 %>% as.integer)
(x <- nd$sort(nd$random$uniform(-1, 1, shape = reticulate::tuple(20 %>% as.integer))))
(x <- nd$array(list(1,-1,1,-1,-1)) %>% nd$sort())
nd$sign(x)
nd$abs(x) %>% nd$argsort()
nd$argsort(x)
(y <- nd$multiply(nd$argsort(x), nd$sign(x)))
nd$argsort(y)
nd$slice_axis(x, axis = 0 %>% as.integer, begin = 0 %>% as.integer, end = 3 %>% as.integer)

nd$slice_axis(x, axis = 0 %>% as.integer, begin = 12 %>% as.integer, end = NULL)
nd$slice_axis(x, axis = 0 %>% as.integer, begin = 0 %>% as.integer, end = 12 %>% as.integer)

(a <- nd$array(list(c(1,2), c(4,5), c(6,7))))
nd$concat(a, nd$array(list(c(-1,1,2))) %>% nd$transpose(), dim = 1 %>% as.integer)
# %%
tree <- NeuralTree()
reticulate::py_len(tree$collect_params()$`_params`)




