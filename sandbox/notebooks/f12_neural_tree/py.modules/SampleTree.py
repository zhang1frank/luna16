# %%

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet.gluon import Block

from collections import OrderedDict

# %%

class Box(Block):
  def __init__(self,
    parent = None, min_list = None, max_list = None, tau = None, **kwargs
  ):
    super(Box, self).__init__(**kwargs)
    with self.name_scope():
      self._parent = parent
      self._min_list = self.params.get("min_list", grad_req = "null", init = mx.init.Constant(min_list))
      self._max_list = self.params.get("max_list", grad_req = "null", init = mx.init.Constant(max_list))
      self._tau = self.params.get("tau", grad_req = "null", init = mx.init.Constant(tau))
      if (min_list is not None):
        self._min_list.set_data(min_list)
        self._min_list.initialize(
          init = mx.init.Constant(min_list)
        )
      if (max_list is not None):
        self._max_list.set_data(max_list)
        self._max_list.initialize(
          init = mx.init.Constant(max_list)
        )
      if (tau is not None):
        self._tau.set_data(tau)
        self._tau.initialize(
          init = mx.init.Constant(tau)
        )
  def forward(self, x):
    parent_tau = 0
    if (self._parent is not None):
      parent_tau = self._parent._box._tau.data()
    delta = self._tau.data() - parent_tau
    s = nd.sum(
      nd.maximum(x - self._max_list.data(), 0) +
      nd.maximum(self._min_list.data() - x, 0),
      axis = 1, keepdims = True
    )
    return 1 - nd.exp(-1 * delta * s)

# %%

class Decision(Block):
  def __init__(self, split, dim, **kwargs):
    super(Decision, self).__init__(**kwargs)
    with self.name_scope():
      self._sharpness = self.params.get("sharpness", init = mx.init.Constant(nd.array([1])))
      self._split = self.params.get("split", init = mx.init.Constant(split))
      self._dim = self.params.get("dim", grad_req = "null", init = mx.init.Constant(dim))
      self._sharpness.set_data(nd.array([1]))
      self._sharpness.initialize(
        init = mx.init.Constant(nd.array([1]))
      )
      self._split.set_data(split)
      self._split.initialize(
        init = mx.init.Constant(split)
      )
      self._dim.set_data(dim)
      self._dim.initialize(
        init = mx.init.Constant(dim)
      )
  def forward(self, x):
    x = nd.pick(x, nd.broadcast_to(self._dim.data(), x.shape[0]), keepdims = True)
    x -= self._split.data()
    x *= nd.relu(self._sharpness.data())
    return nd.tanh(x)

# %%

class Node(Block):
  def __init__(self,
    parent = None, min_list = None, max_list = None, tau = None,
    units = 1, weight_initializer = None, embedding = None,
    decision = None,
    **kwargs
  ):
    super(Node, self).__init__(**kwargs)
    with self.name_scope():
      self._box = Box(
        parent = parent, min_list = min_list, max_list = max_list, tau = tau
      )
      if (decision is not None):
        self._decision = decision()
      if (embedding is not None):
        self._embedding = self.params.get("embedding", init = mx.init.Constant(embedding))
        self._embedding.set_data(embedding)
        self._embedding.initialize(
          init = mx.init.Constant(embedding)
        )
      else:
        self._embedding = self.params.get(
          "embedding", init = weight_initializer,
          shape = (units,)
        )
  def forward(self, x = 0):
    return self._embedding.data()

# %%

class Tree(Block):
  def __init__(self, **kwargs):
    super(Tree, self).__init__(**kwargs)
    with self.name_scope():
      self._structure = OrderedDict()
      self._weightlayer = gluon.contrib.nn.Concurrent()
      self._routerlayer = gluon.contrib.nn.Concurrent()
      self._embeddlayer = gluon.contrib.nn.Concurrent()

      node_0 = Node(
        min_list = nd.array([0, 0]),
        max_list = nd.array([7, 7]),
        tau = nd.array([0.004]),
        decision = lambda: Decision(split = nd.array([3]), dim = nd.array([0]))
      )
      node_1 = Node(
        min_list = nd.array([0, 0]),
        max_list = nd.array([3, 7]),
        tau = nd.array([0.1]),
        decision = lambda: Decision(split = nd.array([2]), dim = nd.array([1]))
      )
      node_2 = Node(
        min_list = nd.array([3, 0]),
        max_list = nd.array([7, 7]),
        tau = nd.array([0.02]),
        decision = lambda: Decision(split = nd.array([4]), dim = nd.array([0]))
      )
      node_3 = Node(
        min_list = nd.array([0, 0]),
        max_list = nd.array([3, 2]),
        tau = nd.array([0.15])
      )
      node_4 = Node(
        min_list = nd.array([0, 2]),
        max_list = nd.array([3, 7]),
        tau = nd.array([0.2])
      )
      node_5 = Node(
        min_list = nd.array([3, 0]),
        max_list = nd.array([4, 7]),
        tau = nd.array([0.2])
      )
      node_6 = Node(
        min_list = nd.array([4, 0]),
        max_list = nd.array([7, 7]),
        tau = nd.array([0.3])
      )

      node_1._box._parent = node_0
      node_2._box._parent = node_0
      node_3._box._parent = node_1
      node_4._box._parent = node_1
      node_5._box._parent = node_2
      node_6._box._parent = node_2

      self._structure[node_0] = {node_1: -1, node_2: 1}
      self._structure[node_1] = {node_3: -1, node_4: 1}
      self._structure[node_2] = {node_5: -1, node_6: 1}
      self._structure[node_3] = None
      self._structure[node_4] = None
      self._structure[node_5] = None
      self._structure[node_6] = None

      self._weightlayer.add(*[x._box for x in iter(self._structure.keys())])
      self._routerlayer.add(*[x._decision for x in iter(self._structure.keys()) if self._structure[x] is not None])
      self._embeddlayer.add(*[x for x in iter(self._structure.keys())])

  def _contextify(self, x):
    psep = self._weightlayer(x)
    splt = self._routerlayer(x)
    embd = self._embeddlayer(x)

    router = nd.zeros_like(psep)
    router_mat = nd.stack(*[nd.zeros_like(psep) for x in self._weightlayer], axis = 1)
    weight = nd.zeros_like(psep)
    embedd = nd.zeros_like(embd)

    def _recurse(node,
      path = nd.zeros_like(splt[:, 0]), prob = nd.ones_like(splt[:, 0]),
      remain = nd.zeros_like(splt[:, 0])
    ):
      children = self._structure[node]

      i_node = next(
        key for key, value in self._weightlayer._children.items()
        if value == node._box
      )
      i_node = int(i_node)

      # calculate the embedd matrix
      embedd[i_node] = node()

      # calculate the router matrix
      if (node._box._parent is not None):
        i = next(
          key for key, value in self._routerlayer._children.items()
          if value == node._box._parent._decision
        )
        i = int(i)
        direction = self._structure[node._box._parent][node]
        path += splt[:, i] * direction - 1

      router[:, i_node] = path + 0.5

      # calculate the weight matrix
      if (node._box._parent is not None and children is not None):
        i_parent = next(
          key for key, value in self._weightlayer._children.items()
          if value == node._box._parent._box
        )
        i_parent = int(i_parent)
        prob *= (1 - psep[:, i_parent])

      if (children is None):
        w = 1 - remain
      else:
        w = psep[:, i_node] * prob
        remain += w

      weight[:, i_node] = w

      # calculate the partial router matrix and weight adj matrix
      path_mat = nd.zeros_like(psep)
      pie = nd.maximum(nd.sign(path + 1), 0)
      cur_node = node
      cur_path = path + 0

      while(1):
        i_cur_node = next(
          key for key, value in self._weightlayer._children.items()
          if value == cur_node._box
        )
        i_cur_node = int(i_cur_node)
        frac = nd.maximum(cur_path + 0.5, -0.5) + 0.5
        path_mat[:, i_cur_node] = frac * pie
        pie -= frac * pie

        if (cur_node._box._parent is not None):
          cur_i = next(
            key for key, value in self._routerlayer._children.items()
            if value == cur_node._box._parent._decision
          )
          cur_i = int(cur_i)
          cur_direction = self._structure[cur_node._box._parent][cur_node]
          cur_path -= splt[:, cur_i] * cur_direction - 1
          cur_node = cur_node._box._parent
        else:
          router_mat[:, i_node, :] = path_mat
          break

      if (children is not None):
        left = next(key for key, value in children.items() if value == -1)
        right = next(key for key, value in children.items() if value == 1)

        _recurse(left, path + 0, prob + 0, remain + 0)
        _recurse(right, path + 0, prob + 0, remain + 0)

      return (
        router,
        router_mat,
        weight,
        embedd
      )

    return _recurse

  def forward(self, x):
    root = next(iter(self._structure.items()))[0]

    router, router_mat, weight, embedd = self._contextify(x)(root)

    presence = nd.sum(router_mat, axis = 2)
    weight_adj = presence * weight
    depth = len(self._weightlayer) - nd.topk(nd.reverse(presence, axis = 1)) - 1
    depth = depth[:, 0]
    remainder = 1 - nd.sum(weight_adj, axis = 1)
    remainder += nd.choose_element_0index(weight_adj, depth)
    weight_adj = nd.fill_element_0index(weight_adj, remainder, depth)

    return (router, weight, weight_adj, router_mat)
    # return nd.sum(nd.expand_dims(weight_adj, axis = 2) * router_mat, axis = 1)

# %%
tree = Tree()
tree.collect_params().initialize(force_reinit = True)



tree(nd.array([[1, 3], [5, 6], [1, 2], [3, 0], [8, 8], [3.5, 3.5]]))

tree(nd.array([[10, 3], [3.5, 4.5], [0, 2], [2.9, 6], [5, 6]]))

nd.sum(nd.expand_dims(weight_adj, axis = 2) * router_mat, axis = 1)

weight_adj

router_mat

nd.sum(router_mat, axis = 2)

(6 - nd.topk(nd.reverse(nd.sum(router_mat, axis = 2), axis = 1)))[:, 0]

nd.choose_element_0index(nd.sum(router_mat, axis = 2), nd.array([4,6,1,0,6]))

weight

nd.sum(router_mat, axis = 2) * weight
nd.sum(nd.sum(router_mat, axis = 2) * weight, axis = 1)

tree.collect_params()

[x for x in tree._weightlayer]

[key for key, value in tree._weightlayer._children.items()]
tree._weightlayer(nd.array([[1, 3], [5, 6], [6, 2], [20, 20]]))
tree._routerlayer(nd.array([[1, 3], [5, 6], [6, 2], [2, 2]]))
tree(nd.array([[1, 3], [5, 6], [6, 2], [3, 0]]))

nd.sum(router_mat, axis = 2)

nd.maximum(nd.sum(router_mat, axis = 2) - weight, 0)

-1 * (nd.maximum(nd.sum(router_mat, axis = 2) - weight, 0) - nd.sum(router_mat, axis = 2))

nd.sum(nd.expand_dims(weight, axis = 2) * router_mat, axis = 1)
nd.sum(nd.sum(nd.expand_dims(weight, axis = 2) * router_mat, axis = 1), axis = 1)

weight
router_mat

# %%

tree.collect_params()

nd.sum(
  nd.maximum(nd.array([[1, 3], [2, 6], [5, 2]]) - nd.array([3, 3]), 0) +
  nd.maximum(nd.array([0, 0]) - nd.array([[1, 3], [2, 6], [5, 2]]), 0),
  axis = 1, keepdims = True
)

# %%
1 - nd.array([2])[0]
next(iter(OrderedDict([("hello", 1)]).items()))
# %%

tree.collect_params()
tree.collect_params()._params["tree5_node0_box0_max_list"]

nd.random.exponential(nd.reciprocal(nd.array([10])))

nd.array([1,2,3])[nd.array([0])]
node_0 = Node(
  min_list = nd.array([0, 0]),
  max_list = nd.array([7, 7]),
  tau = nd.array([0.004]),
  decision = lambda: Decision(split = nd.array([2]), dim = nd.array([1]))
)
node_0._params
node_0.collect_params()

# %%
nd.stack(*[nd.array([1,2,3]), nd.array([4,5,6])], axis = 1)
nd.array([[1,2,3], [4,5,6]])
nd.stack(*[nd.array([.5])], axis= 1)
nd.stack(*[nd.array([[1,1,0], [2,0,0], [3,1,0]]), nd.array([[1,1,0], [5,1,0], [1,0,0]])], axis = 1)
nd.array([[0.1],[0.2]]) * nd.stack(*[nd.array([[1,1,0], [2,0,0], [3,1,0]]), nd.array([[1,1,0], [5,1,0], [1,0,0]])], axis = 1)
a = nd.array([[1,2,3],[4,5,6]])
a[1] = nd.array([2,3,4])
[][0]

# %%

from Tree import Tree
import mxnet as mx
import numpy as np
from mxnet import gluon, nd

tree = Tree(units = 2)

tree.collect_params().initialize(force_reinit = True)

tree.collect_params()

tree._structure

tree(nd.array([[1, 3], [5, 6], [1, 2], [3, 0], [8, 8], [3.5, 3.5]]))

tree._grow(nd.array([[1, 1]]))



# %%

from Node import Node

Node()

node.collect_params()

# %%

a = nd.array([[1,2],[3,4],[-10,-10]])
a
upper = nd.max(a, axis = 0)
lower = nd.min(a, axis = 0)

e = nd.random.exponential(1/nd.sum(upper - lower))

(upper, lower, e)

nd.random.multinomial(nd.array([0.5,0.5]), 10)

# %%

if (nd.sum(nd.array([0,0])) == 0):
  print("yay")

nd.minimum(nd.array([4,5]), nd.array([0, 6]))

