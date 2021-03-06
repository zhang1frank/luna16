# %%

from Node import Node, Box, Decision, Gate

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import Block

from collections import OrderedDict

# %%

class Tree(Block):
  def __init__(self,
    units = 1, weight_initializer = None,
    droprate_init = 0.5, temperature = 0.66, limit_lo = -0.1, limit_hi = 1.1,
    **kwargs):
    super(Tree, self).__init__(**kwargs)

    with self.name_scope():

      def new_gate(**kwargs):
        return Gate(droprate_init, temperature, limit_lo, limit_hi)

      def new_node(**kwargs):
        return Node(
          units = units, weight_initializer = weight_initializer, **kwargs)

      self._new_node = new_node
      self._new_gate = new_gate
      self._structure = OrderedDict([(self._new_node(), None)])
      self._weightlayer = gluon.contrib.nn.Concurrent()
      self._routerlayer = gluon.contrib.nn.Concurrent()
      self._embeddlayer = gluon.contrib.nn.Concurrent()
      self._embeddlayer.add(*[x for x in iter(self._structure.keys())])
      self._weightlayer.add(*[x._box for x in iter(self._structure.keys())])

  def _prune(self, node):
    # a node can only be removed if it has a split
    # when a node is removed, it has its split and box reset
    # also all its children are removed
    # remove from structure, weightlayer, routerlayer, embeddlayer
    # go to its parent, and replace the node with a new node
    # if parent is root, start over I guess

    if (node in self._structure and self._structure[node] is not None):

      def _recurse(node):
        children = self._structure.pop(node)
        i_box = next(
          key for key, value in self._weightlayer._children.items()
          if value == node._box
        )
        i_node = next(
          key for key, value in self._embeddlayer._children.items()
          if value == node
        )
        self._weightlayer._children.pop(i_box)
        self._embeddlayer._children.pop(i_node)

        for i, key in enumerate(list(self._embeddlayer._children.keys())):
          item = self._embeddlayer._children.pop(key)
          self._embeddlayer._children[str(i)] = item

        for i, key in enumerate(list(self._weightlayer._children.keys())):
          item = self._weightlayer._children.pop(key)
          self._weightlayer._children[str(i)] = item

        if (children is not None):
          i_decision = next(
            key for key, value in self._routerlayer._children.items()
            if value == node._decision
          )
          self._routerlayer._children.pop(i_decision)
          for i, key in enumerate(list(self._routerlayer._children.keys())):
            item = self._routerlayer._children.pop(key)
            self._routerlayer._children[str(i)] = item

          left = next(key for key, value in children.items() if value == -1)
          right = next(key for key, value in children.items() if value == 1)

          _recurse(left)
          _recurse(right)

      _recurse(node)

      # replace node in parent reference to fresh node
      with self.name_scope():
        if (node._box._parent is not None):
          p_node = node._box._parent
          dir = self._structure[p_node].pop(node)
          n_node = self._new_node(parent = p_node,
            embedding = p_node._embedding.data())
          self._structure[p_node][n_node] = dir
          self._weightlayer.add(*[n_node._box])
          self._embeddlayer.add(*[n_node])
        else:
          n_node = self._new_node()
          self._structure[n_node] = None
          self._weightlayer.add(*[n_node._box])
          self._embeddlayer.add(*[n_node])

  def _grow(self, x):

    for node in self._embeddlayer._children.values():
      if (hasattr(node, "_decision") and
        (node._decision._gate() == 0 or node._decision._sharpness.data() <= 0)
      ):
        self._prune(node)

    root = next(iter(self._structure.items()))[0]

    def _shard(split, x, l_fn, r_fn):
      splitsortorder = nd.argsort(split, axis = None)
      reorderedx = x[splitsortorder, :]
      reorderedsplit = split[splitsortorder]

      if (reorderedsplit[0] > 0):
        r_fn(reorderedx)
      elif (reorderedsplit[-1] < 0):
        l_fn(reorderedx)
      else:

        splitpt = nd.argsort(reorderedsplit,  axis = 0) * nd.sign(reorderedsplit)
        splitpt = nd.argsort(splitpt, axis = None)[0] + 1
        lx = nd.slice_axis(reorderedx, 0, 0, int(splitpt.asscalar()))
        rx = nd.slice_axis(reorderedx, 0, int(splitpt.asscalar()), None)

        l_fn(lx)
        r_fn(rx)

    def _sample(node):

      def _block(x):
        lower = nd.min(x, axis = 0)
        upper = nd.max(x, axis = 0)
        node._box._init_param("min_list", lower)
        node._box._init_param("max_list", upper)
        extent = nd.sum(upper - lower)

        if (extent > 0):

          with self.name_scope():
            l_node = self._new_node(parent = node,
              embedding = node._embedding.data())
            r_node = self._new_node(parent = node,
              embedding = node._embedding.data())

            self._structure[node] = {l_node: -1, r_node: 1}

            self._weightlayer.add(*[l_node._box, r_node._box])
            self._embeddlayer.add(*[l_node, r_node])

          e = nd.random.exponential(1/extent)
          parent_tau = 0
          if (node._box._parent is not None):
            parent_tau = node._box._parent._box._tau.data()
          node._box._init_param("tau", parent_tau + e)
          dim = nd.random.multinomial((upper - lower)/extent)
          split = nd.random.uniform(lower[dim], upper[dim])

          with node.name_scope():
            node._decision = Decision(split, dim, self._new_gate)
            self._routerlayer.add(*[node._decision])

          decision = node._decision.forward(x, crisp = True)
          _shard(decision, x, _sample(l_node), _sample(r_node))

        else:
          self._structure[node] = None

      return _block

    def _extend(node):

      def _go_above(x, tau):
        lower = nd.min(x, axis = 0)
        lower = nd.minimum(lower, node._box._min_list.data())
        upper = nd.max(x, axis = 0)
        upper = nd.maximum(upper, node._box._max_list.data())

        el = nd.maximum(node._box._min_list.data() - nd.min(x, axis = 0), 0)
        eu = nd.maximum(nd.max(x, axis = 0) - node._box._max_list.data(), 0)
        extent = nd.sum(el + eu)
        dim = nd.random.multinomial((el + eu)/extent)

        btm = el[dim]
        top = eu[dim]
        split = nd.random.multinomial(nd.concat(btm, top, dim = 0)/(btm + top))
        if (split == 0):
          split = nd.random.uniform(lower[dim], node._box._min_list.data()[dim])
        elif (split == 1):
          split = nd.random.uniform(node._box._max_list.data()[dim], upper[dim])

        with self.name_scope():
          p_node = self._new_node(
            parent = node._box._parent,
            min_list = lower,
            max_list = upper,
            tau = tau,
            decision = lambda: Decision(
              split = split, dim = dim, gate = self._new_gate
            ),
            embedding = node._embedding.data()
          )
          s_node = self._new_node(parent = p_node,
            embedding = node._embedding.data())
          node._box._parent = p_node

          if (split < node._box._min_list.data()[dim]):
            # current node is right
            l_node = s_node
            r_node = node

          elif (split > node._box._max_list.data()[dim]):
            # current node is left
            l_node = node
            r_node = s_node

          self._structure[p_node] = {l_node: -1, r_node: 1}

          # p nodes parent also needs to reference p_node and the other child
          if (p_node._box._parent is not None):

            if (self._structure[p_node._box._parent][node] == -1):
              self._structure[p_node._box._parent][p_node] = -1

            elif (self._structure[p_node._box._parent][node] == 1):
              self._structure[p_node._box._parent][p_node] = 1

            self._structure[p_node._box._parent].pop(node)

          elif (p_node._box._parent is None):
            self._structure.move_to_end(p_node, last = False)

          self._weightlayer.add(*[p_node._box, s_node._box])
          self._routerlayer.add(*[p_node._decision])
          self._embeddlayer.add(*[p_node, s_node])

        decision = p_node._decision.forward(x, crisp = True)
        _shard(decision, x, _extend(l_node), _extend(r_node))

      def _go_below(x):
        lower = nd.min(x, axis = 0)
        lower = nd.minimum(lower, node._box._min_list.data())
        upper = nd.max(x, axis = 0)
        upper = nd.maximum(upper, node._box._max_list.data())
        node._box._init_param("min_list", lower)
        node._box._init_param("max_list", upper)

        if (self._structure[node] is not None):
          l_node = next(
            key for key, value in self._structure[node].items()
            if value == -1
          )
          r_node = next(
            key for key, value in self._structure[node].items()
            if value == 1
          )
          decision = node._decision.forward(x, crisp = True)
          _shard(decision, x, _extend(l_node), _extend(r_node))

      def _block(x):
        if (node._box._min_list.shape is None and
          node._box._max_list.shape is None):
          _sample(node)(x)
        else:
          el = nd.maximum(node._box._min_list.data() - nd.min(x, axis = 0), 0)
          eu = nd.maximum(nd.max(x, axis = 0) - node._box._max_list.data(), 0)
          extent = nd.sum(el + eu)

          parent_tau = 0
          if (node._box._parent is not None):
            parent_tau = node._box._parent._box._tau.data()

          if (extent == 0):
            _go_below(x)
          else:
            e = nd.random.exponential(1/extent)
            if (node._box._tau.shape is None or
              parent_tau + e < node._box._tau.data()):
              _go_above(x, parent_tau + e)
            else:
              _go_below(x)

      return _block

    _extend(root)(x)

  def _contextify(self, x):
    if (len(self._routerlayer) > 0):
      psep = self._weightlayer(x)
      splt = self._routerlayer(x)
      embd = self._embeddlayer(x)

      router = nd.zeros_like(psep)
      router_mat = nd.stack(*[
        nd.zeros_like(psep) for x in self._weightlayer
        ], axis = 1)
      weight = nd.zeros_like(psep)
      embedd = nd.zeros_like(embd)

    else:
      return lambda x: self._embeddlayer(x)

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
        # path += splt[:, i] * direction - 1
        path = path + splt[:, i] * direction - 1

      router[:, i_node] = path + 0.5

      # calculate the weight matrix
      if (node._box._parent is not None and children is not None):
        i_parent = next(
          key for key, value in self._weightlayer._children.items()
          if value == node._box._parent._box
        )
        i_parent = int(i_parent)
        # prob *= (1 - psep[:, i_parent])
        prob = prob * (1 - psep[:, i_parent])

      if (children is None):
        w = 1 - remain
      else:
        w = psep[:, i_node] * prob
        # remain += w
        remain = remain + w

      weight[:, i_node] = w

      # calculate the partial router matrix
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
        # pie -= frac * pie
        pie = pie - frac * pie

        if (cur_node._box._parent is not None):
          cur_i = next(
            key for key, value in self._routerlayer._children.items()
            if value == cur_node._box._parent._decision
          )
          cur_i = int(cur_i)
          cur_direction = self._structure[cur_node._box._parent][cur_node]

          # cur_path -= splt[:, cur_i] * cur_direction - 1
          cur_path = cur_path - (splt[:, cur_i] * cur_direction - 1)

          cur_node = cur_node._box._parent
        else:
          router_mat[:, i_node, :] = path_mat
          break

      if (children is not None):
        left = next(key for key, value in children.items() if value == -1)
        right = next(key for key, value in children.items() if value == 1)

        _recurse(left, path + 0, prob + 0, remain + 0)
        _recurse(right, path + 0, prob + 0, remain + 0)

      return (router, router_mat, weight, embedd)

    return _recurse

  def forward(self, x):
    root = next(iter(self._structure.items()))[0]

    if (len(self._routerlayer) > 0):
      router, router_mat, weight, embedd = self._contextify(x)(root)

      presence = nd.sum(router_mat, axis = 2)
      weight_adj = presence * weight
      depth = len(self._weightlayer) - nd.topk(nd.reverse(presence, axis = 1))
      depth -= 1
      depth = depth[:, 0]
      remainder = 1 - nd.sum(weight_adj, axis = 1)
      remainder += nd.choose_element_0index(weight_adj, depth)
      weight_adj = nd.fill_element_0index(weight_adj, remainder, depth)

      head = nd.sum(nd.expand_dims(weight_adj, axis = 2) * router_mat, axis = 1)

      return nd.expand_dims(nd.dot(head, embedd), axis = -1)

    else:
      head = nd.ones_like(nd.slice_axis(x, axis = 1, begin = 0, end = None))
      return self._contextify(x)(root) * head

# %%

from Tree import Tree

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

# %%

tree = Tree()
tree.collect_params().initialize(force_reinit = True)
tree.collect_params()

# %%

root = next(iter(tree._structure.items()))[0]
tree._prune(root)
tree._prune(node)
tree.collect_params()

tree._structure[node]

next(iter(tree._structure.items()))[0]._box._parent
tree._weightlayer
node = tree._embeddlayer._children['1']

tree._routerlayer._children.keys()
tree._weightlayer._children.keys()
tree._embeddlayer._children.keys()

# %%

cost = 0
for decision in tree._routerlayer._children.values():
  gate = decision._gate
  cost += nd.sigmoid(
    gate._qz_loga.data() -
    gate._temperature * nd.log(-1 * gate._limit_lo / gate._limit_hi)
  )
  print(cost)

# %%

[str(i) for i, item in enumerate(tree._embeddlayer._children.items())]

list(tree._embeddlayer._children.keys())

for i, key in enumerate(list(tree._embeddlayer._children.keys())):
  item = tree._embeddlayer._children.pop(key)
  tree._embeddlayer._children[str(i)] = item

# %%

tree._grow(nd.array([[0, 0], [3, 3], [7, 1]]))
tree._grow(nd.array([[10, 10]]))
tree._grow(nd.array([[6, 8]]))
tree._grow(nd.array([[9, 2], [1, 9]]))

tree(nd.array([[1, 1], [2, 2], [-1, -1]]))

# %%

[node._decision._gate() for node in tree._embeddlayer._children.values() if hasattr(node, "_decision")]

# %%

tree._weightlayer(nd.array([[1, 1], [2, 2], [-1, -1]]))
tree._routerlayer(nd.array([[1, 1], [2, 2], [-1, -1]]))

tree._embeddlayer(nd.array([[1, 1], [2, 2], [-1, -1]]))

# %%

root = next(iter(tree._structure.items()))[0]

print(root._box._min_list.data(), root._box._max_list.data(), root._decision._split.data())

# %%

# for box in tree._weightlayer:
#   print(box._min_list.data(), box._max_list.data())
#
# for decision in tree._routerlayer:
#   print(decision._dim.data())
#   print(decision._split.data())

len(tree._routerlayer)

nd.random.uniform(0, 1)

# %%

def traverse(node = next(iter(tree._structure.items()))[0]):
  print("box")
  print((node._box._min_list.data(), node._box._max_list.data()))

  children = tree._structure[node]
  print(type(children))
  if (children is not None):
    print("split")
    print((node._decision._dim.data(), node._decision._split.data()))
    left = next(key for key, value in children.items() if value == -1)
    right = next(key for key, value in children.items() if value == 1)
    print("left")
    traverse(left)
    print("right")
    traverse(right)

traverse()

# %%

root = next(iter(tree._structure.items()))[0]
router, router_mat, weight, embedd = tree._contextify(nd.array([[1, 1], [2, 2], [-1, -1]]))(root)

nd.sum(router_mat, axis = -1)

# %%

root = next(iter(tree._structure.items()))[0]
router_d, router_mat_d, weight_d, embedd_d = tree._contextify(nd.array([[1, 1], [2, 2], [-1, -1]]))(root)

router = nd.stack(*[router_d[key] for key in sorted(router_d)], axis = -1)
weight = nd.stack(*[weight_d[key] for key in sorted(weight_d)], axis = -1)

embedd = nd.stack(*[embedd_d[key] for key in sorted(embedd_d)], axis = 0)
router_mat = nd.stack(*[router_mat_d[key] for key in sorted(router_mat_d)], axis = 1)

where = nd.argmax(nd.maximum(0, 1/(router + 0.5)), axis = 1)

nd.maximum(0, 1/(router + 0.5))

where

presence = nd.sum(router_mat, axis = 2)
weight_adj = presence * weight

weight
router + 0.5
nd.argmin(nd.abs(router + 0.5), axis=1)

router_mat[1][4]

depth

where = nd.argmin(nd.abs(router + 0.5), axis = 1)
nd.concat(*[router_mat[i][k] for i, k in enumerate(where)], dim = 0)

depth = len(tree._weightlayer) - nd.topk(nd.reverse(presence, axis = 1))
depth -= 1
depth = depth[:, 0]
remainder = 1 - nd.sum(weight_adj, axis = 1)
remainder += nd.choose_element_0index(weight_adj, depth)
weight_adj = nd.fill_element_0index(weight_adj, remainder, depth)

head = nd.sum(nd.expand_dims(weight_adj, axis = 2) * router_mat, axis = 1)

print(head)

nd.expand_dims(nd.dot(head, embedd), axis = -1)

old = weight_adj
remainder

depth

old
weight_adj

2 * nd.maximum(0, router)

nd.choose_element_0index(weight_adj, depth)

nd.concat(*[x[d] for d, x in zip(depth, weight_adj)], dim=0)

nd.stack(*[nd.concat(*[y if i != d else r  for i, y in enumerate(x)], dim = 0) for d, r, x in zip(depth, remainder, weight_adj)], axis = 0)

# %%

# tree.collect_params()

next(iter(tree._structure.items()))[0]._decision._gate()

with mx.autograd.record():
  a = next(iter(tree._structure.items()))[0]._decision._gate()

# %%

a = {1: 0, 0: 2, 2: 9}

[a[key] for key in sorted({1: 0, 0: 2, 2: 9})]

sorted({1: 0, 0: 2, 2: 9})

a = nd.stack(nd.array([1,2,3]), nd.array([4,5,6]), axis = -1)
b = nd.stack(nd.array([7,8,9]), nd.array([4,5,6]), axis = -1)

nd.stack(nd.array([5]), nd.array([6]), axis = 0)

nd.stack(a, b, axis = -1)[0]

[x if x == 2 else 0 for x in range(9)]

-1 * nd.log(nd.array([0])) - 5 > 0

