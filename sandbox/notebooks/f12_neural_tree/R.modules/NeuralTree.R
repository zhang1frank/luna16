# %%
magics::.__file__(function(x) {
  x
})

modules::expose(TreeNode,
  module = file.path(.__file__, "TreeNode.R"))

mx <- reticulate::import("mxnet", convert = FALSE)
np <- reticulate::import("numpy", convert = FALSE)
nd <- mx$nd
gluon <- mx$gluon
Block <- mx$gluon$Block

py <- reticulate::import("builtins")

modules::import(magrittr)
modules::import(gsubfn)

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
    "_recurse_node" = function(self, x, node, ...) {
      list(left, right) <- node$`_split_data`(x)
      # if all of data is greater than threshold, send all data to right
      if (left %>% is.null()) {
        self$`_grow_node`(right, node$`_right`, ...)
      # if all of data is less than threshold, send all data to left
      } else if (right %>% is.null()) {
        self$`_grow_node`(left, node$`_left`, ...)
      # determine the highest index belonging to a negative
      # number based on multiplying the argsort of the pre sorted split list
      # with split list sign, that index + 1 is used to slice the data into 2
      } else {
        self$`_grow_node`(left, node$`_left`, ...)
        self$`_grow_node`(right, node$`_right`, ...)
      }
    },
    "_grow_node" = function(self, x, node, ...) {
      if (!(function() {
        # check if node has a split time, if not, don't bother sampling
        if (node$`_tau`$shape %>% reticulate::py_to_r() %>% is.null() ||
          # check if additional data extent == 0, if so, don't bother sampling
          nd$add(
            nd$subtract(node$`_min_list`$data(), x) %>%
              nd$maximum(nd$array(list(0))),
            nd$subtract(x, node$`_max_list`$data()) %>%
              nd$maximum(nd$array(list(0)))
          ) %>% {
              .$asnumpy()
            }
            == 0
        # invoke extend node with current node
      ) FALSE else self$`_extend_ node`(x, node, ...)
        # check to make sure if node was actually extended
      }
      )()) {
        # if extend node not extended
        # update the node boundaries with new data
        node$`_update_extent`(x)
        # if no split time, invoke split node
        if (node$`_tau`$shape %>% is.null()) {
          self$`_split_node`(x, node, ...)
        # if there is split time, recurse on children
        } else {
          self$`_recurse_node`(x, node, ...)
        }
      }
    },
    "_split_node" = function(self, x, node, ...) {
      # check to split or not
      # if don't split, add node to leaf layer, end
      # if split
      if (self$`_stop_split`(x, node, ...)) {
        # check if already in leaf layer, add if not yet there
        self$`_leaf_layer`$add(node)
      } else {
        # determine split time based on node boundaries
        node$`_update_tau`()
        # sample feature
        # sample split
        # determine best split
        list(dim, split) <- self$`_draw_feature`() %>%
          self$`_draw_split`() %>%
          self$`_pick_best_split`()
        # set split, change forward based on dim, add node to decision layer
        node$`_create_split`(dim, split)
        self$`_decision_layer`$add(node)
        # create children nodes, blanks
        reticulate::py_set_attr(node, "_left", TreeNode(parent = node))
        reticulate::py_set_attr(node, "_right", TreeNode(parent = node))
        # promote current node to decision node
        # check if node is in leaf layer, need to pop from there if so

        # shard the data based on the split
        # invoke grow node with left node
        # invoke grow node with right node
        self$`_recurse_node`(x, node, ...)
      }
    },
    "_extend_node" = function(self, x, node, ...) {
      # return true if node is extended
      # determine additional data extent compared to node
      extent_l <- nd$sub(node$`_min_list`,
        nd$min(x, axis = 0 %>% as.integer)) %>%
        nd$broadcast_maximum(nd$array(list(0)))
      extent_u <- nd$sub(nd$max(x, axis = 0 %>% as.integer),
        node$`_max_list`) %>%
        nd$broadcast_maximum(nd$array(list(0)))
      # if node has no parent, then time split of parent = 0
      # sample time based on additional data extent
      E <- nd$add(extent_l, extent_u) %>% nd$sum() %>% nd$reciprocal() %>%
      nd$random$exponential() %>% {
        if (self$`_parent` %>% is.null()) {
          return(.)
        } else {
          nd$add(., self$`_parent`$`_tau`$data())
        }
      }
      # if (split of parent + sampled time < current node split time)
      if (E < node$`_tau`$data()) {
        # sample feature
        # sample split
        # determine best split
        list(dim, split) <- self$`_draw_feature`() %>%
          self$`_draw_split`() %>%
          self$`_pick_best_split`()
        TRUE
      } else {
        FALSE
      }
        # create a new parent node with min, max,
          # previous parent must now have new parent as child
          # new parent has previous parent as parent
          # new parent has node as the appropriate child based on axis and
          # whether bounds are both greater than or less than parent split
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
        # if you want to bootstrap, or do some other form of sample selection
        self$`_grow_node`(self$`_draw_data`(x, ...), self$`_root_node`)
      }
      x %>% self$decision_layer() %>% self$leaf_layer()
    }
  )
)

# %%
modules::export("NeuralTree")