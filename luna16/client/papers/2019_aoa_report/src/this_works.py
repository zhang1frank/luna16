# %%

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import Block, HybridBlock
from mxnet import init
from mxnet.gluon import nn, contrib

# %%

class Node(HybridBlock):

  def __init__(self, center, radius=0, decision=None, left=None, right=None):
    super(Node, self).__init__()
    center = nd.array(center)
    radius = nd.array(radius)
    with self.name_scope():
      self.center = self.params.get(
        'center',
        shape=center.shape[-1],
        init=init.Constant(center),
        grad_req='null'
      )
      self.radius = self.params.get(
        'radius',
        shape=1,
        init=init.Constant(radius),
        grad_req='null'
      )

    self.collect_params().initialize()
    self.child = {'decision': decision, 'left': left, 'right': right}

  def hybrid_forward(self, F, x, center, radius):
    pass

# %%

class Split(HybridBlock):

  def __init__(self, weight, bias, sharpness, tau, decision, side):
    super(Split, self).__init__()
    weight = nd.array(weight)
    bias = nd.array(bias)
    sharpness = nd.array(sharpness)
    tau = nd.array(tau)
    with self.name_scope():
      self.sharpness = self.params.get(
        'sharpness',
        shape=1,
        init=init.Constant(sharpness)
      )
      self.tau = self.params.get(
        'tau',
        shape=1,
        init=init.Constant(tau),
        grad_req='null'
      )
      self.split = nn.Dense(
        units=1,
        in_units=weight.shape[-1],
        weight_initializer=init.Constant(weight),
        bias_initializer=init.Constant(bias)
      )

    self.collect_params().initialize()
    self.parent = {'decision': decision, 'side': side}

  def hybrid_forward(self, F, x, sharpness, tau):
    x = self.split(x)
    a = F.log10(1 + F.exp(sharpness))
    return F.sigmoid(a * x)

# %%

class Leaf(HybridBlock):

  def __init__(self, layer, node=None, decision=None, side=None):
    super(Leaf, self).__init__()
    with self.name_scope():
      self.layer = layer

    self.parent = {'node': node, 'decision': decision, 'side': side}

  def hybrid_forward(self, F, x):
    return self.layer(x)

# %%

class Tree(HybridBlock):

  def __init__(self, layer_initializer):
    super(Tree, self).__init__()
    self.layer_initializer = layer_initializer
    with self.name_scope():
      self.nodes = contrib.nn.HybridConcurrent()
      self.splits = contrib.nn.HybridConcurrent()
      self.leaves = contrib.nn.HybridConcurrent()
      self.leaves.add(Leaf(self.layer_initializer()))

  def hybrid_forward(self, F, x):
    leaf_output = []
    leaf_weight = []
    for leaf in self.leaves:
      leaf_path = [F.ones((x.shape[0], 1))]
      split = leaf.parent
      while 1:
        if split['decision'] is None: break
        leaf_path.append(
          F.abs(0**split['side'] - split['decision'](x))
        )
        split = split['decision'].parent

      leaf_weight.append(F.prod(F.stack(*leaf_path), axis=0))
      leaf_output.append(leaf(x))

    # return leaf_weight, leaf_output
    return F.sum(
      F.stack(*leaf_weight, axis=-1) * F.stack(*leaf_output, axis=-1),
      axis=-1
    )

# %%

def extend_tree(x, tree):

  def recurse(x, node, p_tau):
    n = 1
    mean = node.center.data()
    var = (0.5 * node.radius.data()) ** 2
    N = x.shape[0]
    x_mean = nd.mean(x, axis=0)
    x_var = (N ** -1) * nd.sum((x - x_mean) ** 2, axis=0)
    z_mean = (n * mean + N * x_mean) / (n + N)
    z_var = ((n * (mean + var) + N * (x_mean + x_var)) / (n + N)) - z_mean
    z_radius = 2 * (nd.max(z_var) ** 0.5)
    node.center.set_data(z_mean)
    node.radius.set_data(z_radius)
    if node.child['decision'] is None:
      leaf = next(l for l in tree.leaves if l.parent['node'] == node)
      tesselate(x, leaf, p_tau)
      return

    E = nd.random.exponential(z_radius ** -1)
    node.child['decision'].tau.set_data(p_tau + E)
    split = node.child['decision']
    side = nd.sign(split.split(x))
    order = nd.argsort(side, axis = None)
    x = x[order, :]
    side = side[order, :]
    if side[0] > 0:
      recurse(x, node.child['right'], split.tau.data())
    elif side[-1] < 0:
      recurse(x, node.child['left'], split.tau.data())
    else:
      orderside = nd.argsort(side, axis=0) * side
      cutpt = nd.argsort(orderside, axis=None, dtype='int32')[0].asscalar() + 1
      x_l = x[0:cutpt]
      x_r = x[cutpt:None]
      recurse(x_l, node.child['left'], split.tau.data())
      recurse(x_r, node.child['right'], split.tau.data())

  def tesselate(x, leaf, p_tau):
    if (len(x) < 2):
      return

    add_split(x, leaf, p_tau)
    split = leaf.parent['decision']
    node = leaf.parent['node']
    side = nd.sign(split.split(x))
    order = nd.argsort(side, axis = None)
    x = x[order, :]
    side = side[order, :]
    orderside = nd.argsort(side, axis=0) * side
    cutpt = nd.argsort(orderside, axis=None, dtype='int32')[0].asscalar() + 1
    x_l = x[0:cutpt]
    x_r = x[cutpt:None]
    leaf.parent['side'] = 0
    new_leaf = Leaf(
      layer=tree.layer_initializer(),
      node=leaf.parent['node'],
      decision=leaf.parent['decision'],
      side=1
    )
    tree.leaves.add(new_leaf)
    add_node(x_l, leaf)
    add_node(x_r, new_leaf)
    node.child['left'] = leaf.parent['node']
    node.child['right'] = new_leaf.parent['node']
    tesselate(x_l, leaf, split.tau.data())
    tesselate(x_r, new_leaf, split.tau.data())

  def add_split(x, leaf, p_tau):
    center = leaf.parent['node'].center.data()
    radius = leaf.parent['node'].radius.data()
    tau = p_tau + nd.random.exponential(radius ** -1)
    while 1:
      s = nd.random.normal(shape=(2, x.shape[-1]))
      s = s / nd.norm(s, axis=-1, keepdims=True)
      r = nd.random.uniform(low=nd.array([0]), high=radius)
      r = r * nd.random.uniform() ** (1/3)
      if nd.sign(s[0][-1]) > 0:
        weight = s[0]
        bias = nd.dot(s[0], -1 * r * (s[1] + center))
        y = nd.sign(nd.dot(x, weight) + bias)
        if nd.abs(nd.sum(y)) != len(y):
          break

    split = Split(
      weight=weight,
      bias=bias,
      sharpness=3 / radius,
      tau=tau,
      decision=leaf.parent['decision'],
      side=leaf.parent['side']
    )
    tree.splits.add(split)
    leaf.parent['node'].child['decision'] = split
    leaf.parent['decision'] = split

  def add_node(x, leaf):
    N = x.shape[0]
    x_mean = nd.mean(x, axis=0)
    x_var = (N ** -1) * nd.sum((x - x_mean) ** 2, axis=0)
    x_radius = 2 * (nd.max(x_var) ** 0.5)
    node = Node(x_mean, x_radius)
    tree.nodes.add(node)
    leaf.parent['node'] = node

  with tree.name_scope():
    if len(tree.nodes) == 0:
      leaf = tree.leaves[0]
      add_node(x, leaf)
      tesselate(x, leaf, 0)

    else:
      recurse(x, tree.nodes[0], 0)

# %%

import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.utils import shuffle
import numpy as np
from mxnet import autograd

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

# X, Y = sklearn.datasets.make_circles(200, factor=.3, noise=.05)
X, Y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=Y, cmap=plt.cm.Spectral)

def plot_decision_boundary(pred_func, X, Y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Spectral)

# %%

subnet = gluon.nn.Sequential()
with subnet.name_scope():
  subnet.add(nn.Dense(5, activation = "tanh"))

subnet.collect_params().initialize(mx.init.Uniform(1), ctx=model_ctx)

tree = Tree(lambda: nn.Dense(1))
extend_tree(subnet(nd.array(X[0:9])), tree)

net = gluon.nn.Sequential()
with net.name_scope():
  net.add(subnet)
  net.add(tree)

net.collect_params().initialize(mx.init.Uniform(1), ctx=model_ctx)
error = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

X, Y = shuffle(X, Y)
for x, y in zip(np.split(X, 10), np.split(Y, 10)):
  data = nd.array(x).as_in_context(model_ctx)
  label = nd.array(y).as_in_context(model_ctx)
  with autograd.record():
    output = net(data)
    loss = error(output, label)

  loss.backward()
  trainer.step(data.shape[0])

plot_decision_boundary(lambda x: net(nd.array(x)).sigmoid()[:, 0].asnumpy(), X, Y)

net.collect_params()

# %%

net = gluon.nn.Sequential()
with net.name_scope():
  net.add(nn.Dense(10, activation = "tanh"))
  net.add(nn.Dense(10, activation = "tanh"))
  net.add(nn.Dense(1))

net.collect_params().initialize(mx.init.Uniform(1), ctx=model_ctx)
error = gluon.loss.SigmoidBinaryCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

for _ in range(1000):
  X, Y = shuffle(X, Y)
  for x, y in zip(np.split(X, 10), np.split(Y, 10)):
    data = nd.array(x).as_in_context(model_ctx)
    label = nd.array(y).as_in_context(model_ctx)
    with autograd.record():
      output = net(data)
      loss = error(output, label)

    loss.backward()
    trainer.step(data.shape[0])

plot_decision_boundary(lambda x: net(nd.array(x)).sigmoid()[:, 0].asnumpy(), X, Y)

# %%

a = Tree(lambda: nn.Dense(2))
a.collect_params().initialize()

data = nd.array([[1, 0], [2, 2], [-1, -5]])
a(data)

extend_tree(data, a)

a.collect_params().initialize()
a(data)

data = nd.array([[5, 0], [1, 2], [5, 6], [0, 0], [10, 10]])

extend_tree(data, a)

a.collect_params()

a.collect_params().initialize()
a(data)

for l in a.leaves:
  print(l.parent)

for s in a.splits:
  print(s.parent)

for n in a.nodes:
  print(n.child)