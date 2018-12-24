from Node import Node, Box, Decision, Gate

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import Block

from collections import OrderedDict

class Tree(Block):
  def __init__(self,
    units = 1, weight_initializer = None,
    droprate_init = 0.1, temperature = 0.66, limit_lo = -0.1, limit_hi = 1,
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
          n_node = self._new_node(parent = p_node)
          self._structure[p_node][n_node] = dir
          self._structure[n_node] = None
          self._weightlayer.add(*[n_node._box])
          self._embeddlayer.add(*[n_node])
        else:
          n_node = self._new_node()
          self._structure[n_node] = None
          self._weightlayer.add(*[n_node._box])
          self._embeddlayer.add(*[n_node])

  def _grow(self, x):

    # for node in list(self._embeddlayer._children.values()):
    #   if (hasattr(node, "_decision") and
    #     (node._decision._gate() == 0 or node._decision._sharpness.data() <= 0)
    #   ):
    #     self._prune(node)

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
            l_node = self._new_node(parent = node)
            r_node = self._new_node(parent = node)

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
            )
          )
          s_node = self._new_node(parent = p_node)
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
            if (parent_tau + e < node._box._tau.data()):
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

      # router = nd.zeros_like(psep)
      # router_mat_t = nd.stack(*[
      #   nd.zeros_like(psep) for x in self._weightlayer
      #   ], axis = 1)
      # weight = nd.zeros_like(psep)
      # embedd = nd.zeros_like(embd)
      router = {}
      router_mat = {}
      weight = {}
      embedd = {}

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
      # embedd[i_node] = node()
      embedd[i_node] = node()

      # calculate the router matrix
      if (node._box._parent is not None):
        i = next(
          key for key, value in self._routerlayer._children.items()
          if value == node._box._parent._decision
        )
        i = int(i)
        direction = self._structure[node._box._parent][node]
        path = path + splt[:, i] * direction - 1

      # router[:, i_node] = path + 0.5
      router[i_node] = path + 0.5

      # prevent routing decay
      # path = nd.minimum(0, nd.sign(path + 1))

      # calculate the weight matrix
      if (node._box._parent is not None and children is not None):
        i_parent = next(
          key for key, value in self._weightlayer._children.items()
          if value == node._box._parent._box
        )
        i_parent = int(i_parent)
        prob = prob * (1 - psep[:, i_parent])

      if (children is None):
        w = 1 - remain
      else:
        w = psep[:, i_node] * prob
        remain = remain + w

      # weight[:, i_node] = w
      weight[i_node] = w

      # calculate the partial router matrix
      # path_mat_t = nd.zeros_like(psep)
      path_mat = {}
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
        # path_mat_t[:, i_cur_node] = frac * pie
        path_mat[i_cur_node] = frac * pie
        pie = pie - frac * pie

        if (cur_node._box._parent is not None):
          cur_i = next(
            key for key, value in self._routerlayer._children.items()
            if value == cur_node._box._parent._decision
          )
          cur_i = int(cur_i)
          cur_direction = self._structure[cur_node._box._parent][cur_node]
          cur_path = cur_path - (splt[:, cur_i] * cur_direction - 1)
          cur_node = cur_node._box._parent
        else:
          # router_mat_t[:, i_node, :] = path_mat_t
          n_node = len(self._weightlayer)
          router_mat[i_node] = nd.stack(
            *[path_mat[key]
              if key in path_mat else nd.zeros_like(splt[:, 0])
              for key in range(n_node)],
            axis = -1
          )

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
      router_d, router_mat_d, weight_d, embedd_d = self._contextify(x)(root)

      router = nd.stack(*[router_d[key] for key in sorted(router_d)], axis = -1)
      weight = nd.stack(*[weight_d[key] for key in sorted(weight_d)], axis = -1)

      embedd = nd.stack(*[embedd_d[key] for key in sorted(embedd_d)], axis = 0)
      router_mat = nd.stack(
        *[router_mat_d[key] for key in sorted(router_mat_d)], axis = 1)

      presence = nd.sum(router_mat, axis = 2)
      weight_adj = presence * weight
      depth = len(self._weightlayer) - nd.topk(nd.reverse(presence, axis = 1))
      depth = depth - 1
      depth = depth[:, 0]
      remainder = 1 - nd.sum(weight_adj, axis = 1)
      # remainder = remainder + nd.choose_element_0index(weight_adj, depth)
      remainder = remainder + nd.concat(
        *[x[d] for d, x in zip(depth, weight_adj)], dim = 0)
      # weight_adj = nd.fill_element_0index(weight_adj, remainder, depth)
      weight_adj = nd.stack(
        *[nd.concat(*[y if i != d else r for i, y in enumerate(x)], dim = 0)
            for d, r, x in zip(depth, remainder, weight_adj)
          ], axis = 0)

      head = nd.sum(nd.expand_dims(weight_adj, axis = 2) * router_mat, axis = 1)

      return nd.dot(head, embedd)

    else:
      head = nd.ones_like(nd.slice_axis(x, axis = 1, begin = 0, end = None))
      return self._contextify(x)(root) * head