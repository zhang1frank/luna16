# %%

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import Block

from collections import OrderedDict

# %%

class Custom_Block(Block):
  def __init__(self, **kwargs):
    super(Custom_Block, self).__init__(**kwargs)

  def _init_param(self, param_name, value):
    param =  self.params.get(param_name)
    if (value is not None and param.shape is None):
      param.set_data(value)
      param.initialize(init = mx.init.Constant(value))
    elif (value is not None):
      param.set_data(value)

# %%

class Gate(Custom_Block):
  def __init__(self, droprate_init, temperature, limit_lo, limit_hi, **kwargs):
    super(Gate, self).__init__(**kwargs)
    self._temperature = nd.array([temperature])
    self._limit_lo = nd.array([limit_lo])
    self._limit_hi = nd.array([limit_hi])

    droprate_init = nd.array([droprate_init])
    qz_loga = nd.log(1 - droprate_init) - nd.log(droprate_init)

    with self.name_scope():
      self._qz_loga = self.params.get(
        "qz_loga", init = mx.init.Constant(qz_loga),
        allow_deferred_init = True)
      self._init_param("qz_loga", qz_loga)

  def forward(self, x = 0):
    if (mx.autograd.is_training()):
      u = nd.random.uniform(0, 1)
      s = nd.log(u) - nd.log(1 - u) + self._qz_loga.data()
      if (self._temperature == 0):
        s = nd.sign(s)
      else:
        s = nd.sigmoid(s / self._temperature)

    else:
      s = nd.sigmoid(self._qz_loga.data())

    s = s * (self._limit_hi - self._limit_lo) + self._limit_lo

    return nd.minimum(1, nd.maximum(s, 0))

# %%

class Box(Custom_Block):
  def __init__(self,
    parent = None, min_list = None, max_list = None, tau = None, **kwargs
  ):
    super(Box, self).__init__(**kwargs)

    inf = -1 * nd.log(nd.array([0]))

    with self.name_scope():
      self._parent = parent
      self._min_list = self.params.get(
        "min_list", grad_req = "null", init = mx.init.Constant(min_list),
        allow_deferred_init = True)
      self._max_list = self.params.get(
        "max_list", grad_req = "null", init = mx.init.Constant(max_list),
        allow_deferred_init = True)
      self._tau = self.params.get(
        "tau", grad_req = "null",
        init = mx.init.Constant(tau if tau is not None else inf),
        allow_deferred_init = True)

      self._init_param("min_list", min_list)
      self._init_param("max_list", max_list)
      self._init_param("tau", tau if tau is not None else inf)

  def forward(self, x):
    parent_tau = 0
    if (self._parent is not None):
      parent_tau = self._parent._box._tau.data()

    if (self._min_list.shape is None and self._max_list.shape is None):
      return nd.expand_dims(nd.ones_like(x[:, 0]), axis = -1)

    s = nd.sum(
      nd.maximum(x - self._max_list.data(), 0) +
      nd.maximum(self._min_list.data() - x, 0),
      axis = 1, keepdims = True
    )
    delta = self._tau.data() - parent_tau

    return 1 - nd.exp(-1 * delta * s)

# %%

class Decision(Custom_Block):
  def __init__(self, split, dim, gate, **kwargs):

    super(Decision, self).__init__(**kwargs)
    with self.name_scope():
      self._gate = gate()
      self._sharpness = self.params.get(
        "sharpness", init = mx.init.Constant(nd.array([1])),
        allow_deferred_init = True)
      self._split = self.params.get(
        "split", init = mx.init.Constant(split),
        allow_deferred_init = True)
      self._dim = self.params.get(
        "dim", grad_req = "null", init = mx.init.Constant(dim),
        allow_deferred_init = True)

      self._init_param("sharpness", nd.array([1]))
      self._init_param("split", split)
      self._init_param("dim", dim)

  def forward(self, x, crisp = False):
    pick_index = nd.broadcast_to(self._dim.data(), x.shape[0])
    x = nd.pick(x, pick_index, keepdims = True)
    x = x - self._split.data()
    if (crisp == False):
      x = x * nd.relu(self._sharpness.data())

    return nd.sigmoid(x)

# %%

class Node(Custom_Block):
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
        self._embedding = self.params.get(
          "embedding", init = mx.init.Constant(embedding))
        self._init_param("embedding", embedding)
      else:
        self._embedding = self.params.get(
          "embedding", init = weight_initializer,
          shape = (units,)
        )
        self._embedding.initialize()
  def forward(self, x = 0):
    return self._embedding.data()

# %%

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
      split = 2 * split - 1
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
      splt = self._routerlayer(x)
      embd = self._embeddlayer(x)

      router = {}
      embedd = {}

    else:
      return lambda x: self._embeddlayer(x)

    def _recurse(node,
      path = nd.ones_like(splt[:, 0])
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
        weight = 0.5 * (direction * (2 * splt[:, i] - 1) + 1)
        if (children is None):
          router[i_node] = path * weight
        else:
          gate = node._decision._gate()
          router[i_node] = path * weight * (1 - gate)
          path = path * weight - router[i_node]
      else:
        if (children is None):
          router[i_node] = path
        else:
          gate = node._decision._gate()
          router[i_node] = path * (1 - gate)
          path = path - router[i_node]

      if (children is not None):
        left = next(key for key, value in children.items() if value == -1)
        right = next(key for key, value in children.items() if value == 1)

        _recurse(left, path + 0)
        _recurse(right, path + 0)

      return (router, embedd)

    return _recurse

  def forward(self, x):
    root = next(iter(self._structure.items()))[0]

    if (len(self._routerlayer) > 0):
      router_d, embedd_d = self._contextify(x)(root)

      embedd = nd.stack(*[embedd_d[key] for key in sorted(embedd_d)], axis = 0)
      router = nd.stack(*[router_d[key] for key in sorted(router_d)], axis = -1)

      return nd.dot(router, embedd)

    else:
      head = nd.ones_like(nd.slice_axis(x, axis = 1, begin = 0, end = None))
      return self._contextify(x)(root) * head

# %%

tree = Tree()
tree.collect_params().initialize(force_reinit = True)
tree.collect_params()

# %%

tree._grow(nd.array([[0, 0], [3, 3], [7, 1]]))

# %%

tree.collect_params()

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
tree._contextify(nd.array([[3, 3]]))(root)

tree._routerlayer(nd.array([[3, 3]]))

root._decision._gate()

# %%

router_d = tree._contextify(nd.array([[3, 3]]))(root)[0]
embedd_d = tree._contextify(nd.array([[3, 3]]))(root)[1]

embedd = nd.stack(*[embedd_d[key] for key in sorted(embedd_d)], axis = 0)
router = nd.stack(*[router_d[key] for key in sorted(router_d)], axis = -1)


nd.dot(router, embedd)

tree._routerlayer(nd.array([[3, 3]]))

nd.sigmoid(nd.array([6.0056-3]))

# %%

import matplotlib.pyplot as plt
import numpy as np

import sklearn.datasets
from sklearn.utils import shuffle

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

# %%

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis = 0)
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y = np.sin(X).ravel()
# y[::2] += 3 * (0.5 - rng.rand(40))

# %%

plt.figure()
plt.scatter(X, y, s = 20, edgecolor = "black", c = "darkorange", label = "data")

plt.show()

# %%

tree = Tree(droprate_init = 0.01)
tree.collect_params().initialize(force_reinit = True)
error = gluon.loss.L2Loss()

# %%

X, y = shuffle(X, y)
data = nd.array(X[:20])
target = nd.array(y[:20])
tree._grow(data)

X, y = shuffle(X, y)
data = nd.array(X[:20])
target = nd.array(y[:20])

less = tree.collect_params()
for key in list(less._params.keys()):
  if less[key].shape is None:
    less._params.pop(key)

trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 1})

root = next(iter(tree._structure.items()))[0]
before = root._decision._gate()

# %%

X, y = shuffle(X, y)
data = nd.array(X[:20])
target = nd.array(y[:20])
tree._grow(data)

# %%

less = tree.collect_params()
for key in list(less._params.keys()):
  if less[key].shape is None:
    less._params.pop(key)

trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 3})

with mx.autograd.record():
  # cost = []
  # for decision in tree._routerlayer._children.values():
  #   gate = decision._gate
  #   cost.append(nd.sigmoid(
  #     gate._qz_loga.data() -
  #     gate._temperature * nd.log(-1 * gate._limit_lo / gate._limit_hi)
  #   ))
  #
  # cost = nd.sum(nd.stack(*cost))
  loss = error(tree(nd.array(data)), nd.array(target))
  # loss = loss + 0.01*cost

loss.backward()
trainer.step(data.shape[0], ignore_stale_grad = True)

# %%

after = [node._decision._gate().asscalar() if hasattr(node, "_decision") else None for node in tree._embeddlayer._children.values()]

if (len(tree._embeddlayer) > 1):
  for node, value in zip(list(tree._embeddlayer._children.values()), after):
    if (value is None):
        pass
    elif (value <= before):
      if node == root:
        pass
      else:
        tree._prune(node)

less = tree.collect_params()
for key in list(less._params.keys()):
  if less[key].shape is None:
    less._params.pop(key)

trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 1})

with mx.autograd.record():
  loss = error(tree(nd.array(data)), nd.array(target))

loss.backward()
trainer.step(data.shape[0], ignore_stale_grad = True)

print(len(tree._routerlayer))

# %%

def traverse(node = next(iter(tree._structure.items()))[0]):
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

plt.figure()
plt.scatter(X, y, s = 20, edgecolor = "black", c = "darkorange", label = "data")
y_1 = tree(nd.array(X_test)).asnumpy()
plt.plot(X_test, y_1, color = "cornflowerblue", linewidth = 2)

plt.show()

# %%

X, y = shuffle(X, y)

for data, target in zip(np.split(X, 10), np.split(y, 10)):

  tree._grow(nd.array(data))

  less = tree.collect_params()
  for key in list(less._params.keys()):
    if less[key].shape is None:
      less._params.pop(key)

  trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 1})

  with mx.autograd.record():
    loss = error(tree(nd.array(data)), nd.array(target))

  loss.backward()
  trainer.step(data.shape[0], ignore_stale_grad = True)

  after = [node._decision._gate().asscalar() if hasattr(node, "_decision") else None for node in tree._embeddlayer._children.values()]

  if (len(tree._embeddlayer) > 1):
    mode = max(set([x for x in after if x is not None]), key = after.count)
    after.count(mode)
    hit_value = mode if after.count(mode) > 1 else None

    # hit_value = max(set([x for x in after if x is not None]), key = after.count)

    for node, value in zip(list(tree._embeddlayer._children.values()), after):
      if (value == hit_value):
        tree._prune(node)

  print(len(tree._routerlayer))

# %%

X, y = sklearn.datasets.make_circles(200, factor=.3, noise=.05)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# %%

tree = Tree(droprate_init = 0.01)

net = gluon.nn.Sequential()

with net.name_scope():
  # prenet = gluon.nn.Dense(5, activation = "relu")
  # net.add(prenet)
  net.add(tree)
  net.add(gluon.nn.Dense(2))

net.collect_params().initialize(mx.init.Normal(sigma = 0.1), force_reinit = True, ctx = model_ctx)
error = gluon.loss.SoftmaxCrossEntropyLoss()

# %%

X, y = shuffle(X, y)

data = nd.array(X[:20])
target = nd.array(y[:20])

data = nd.array(data).as_in_context(model_ctx)
target = nd.array(target).as_in_context(model_ctx)

tree._grow(prenet(data))

# %%

X, y = shuffle(X, y)

data = nd.array(X[:20])
target = nd.array(y[:20])

data = nd.array(data).as_in_context(model_ctx)
target = nd.array(target).as_in_context(model_ctx)

less = net.collect_params()
for key in list(less._params.keys()):
  if less[key].shape is None:
    less._params.pop(key)

trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 2})

with mx.autograd.record():
  loss = error(net(data), target)
  # cost = []
  # for decision in tree._routerlayer._children.values():
  #   gate = decision._gate
  #   cost.append(nd.sigmoid(
  #     gate._qz_loga.data() -
  #     gate._temperature * nd.log(-1 * gate._limit_lo / gate._limit_hi)
  #   ))
  #
  # cost = nd.sum(nd.stack(*cost))
  # loss = loss = error(net(data), target)
  # loss = 2*loss + 2*cost

loss.backward()
trainer.step(data.shape[0])

# %%

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.05
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# %%

plot_decision_boundary(lambda x: nd.argmax(net(nd.array(x)), axis = 1).asnumpy())

# %%

net(nd.array(data))

target

# %%

prenet.weight.data()

# %%

prenet.weight.data()