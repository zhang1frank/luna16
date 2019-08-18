# %%

import matplotlib.pyplot as plt

from Tree import Tree

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

import sklearn.datasets
from sklearn.utils import shuffle

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

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

X, y = sklearn.datasets.make_circles(200, factor=.3, noise=.05)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# %%

tree = Tree(droprate_init = 0.01)

net = gluon.nn.Sequential()

with net.name_scope():
  prenet = gluon.nn.Dense(5, activation = "tanh")
  net.add(prenet)
  net.add(tree)
  net.add(gluon.nn.Dense(2))

net.collect_params().initialize(mx.init.Normal(sigma = 0.1), force_reinit = True, ctx = model_ctx)
error = gluon.loss.SoftmaxCrossEntropyLoss()

# %%

X, y = shuffle(X, y)

for data, target in zip(np.split(X, 10), np.split(y, 10)):

  data = nd.array(data).as_in_context(model_ctx)
  target = nd.array(target).as_in_context(model_ctx)

  tree._grow(prenet(data))
  # tree._grow(data)

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

  after = [node._decision._gate().asscalar() if hasattr(node, "_decision") else None for node in tree._embeddlayer._children.values()]
  size = len(after)

  if (len(tree._embeddlayer) > 1):
    mode = max(set([x for x in after if x is not None]), key = after.count)
    after.count(mode)
    hit_value = mode if after.count(mode) > 1 else None

    # hit_value = max(set([x for x in after if x is not None]), key = after.count)

    for node, value in zip(list(tree._embeddlayer._children.values()), after):
      if (value == hit_value or value == 0):
        tree._prune(node)

  print(len(tree._routerlayer))

print("done")

# %%

def traverse(node = next(iter(tree._structure.items()))[0]):
  print("box")
  print((node._box._min_list.data() if node._box._min_list.shape is not None else None, node._box._max_list.data() if node._box._max_list.shape is not None else None))

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

tree._grow(prenet(data))

less = net.collect_params()
for key in list(less._params.keys()):
  if less[key].shape is None:
    less._params.pop(key)

trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 0.1})

with mx.autograd.record():
  # loss = error(net(data), target)
  cost = []
  for decision in tree._routerlayer._children.values():
    gate = decision._gate
    cost.append(nd.sigmoid(
      gate._qz_loga.data() -
      gate._temperature * nd.log(-1 * gate._limit_lo / gate._limit_hi)
    ))

  cost = nd.sum(nd.stack(*cost))
  loss = error(tree(nd.array(data)), nd.array(target))
  loss = 2*loss + 2*cost

loss.backward()
trainer.step(data.shape[0], ignore_stale_grad = True)

after = [node._decision._gate().asscalar() if hasattr(node, "_decision") else None for node in tree._embeddlayer._children.values()]
size = len(after)

if (len(tree._embeddlayer) > 1):
  mode = max(set([x for x in after if x is not None]), key = after.count)
  after.count(mode)
  hit_value = mode if after.count(mode) > 1 else None

  # hit_value = max(set([x for x in after if x is not None]), key = after.count)

  for node, value in zip(list(tree._embeddlayer._children.values()), after):
    if (value == hit_value or value == 0):
      tree._prune(node)

print(len(tree._routerlayer))

# %%

prenet(data)
net(data)

prenet.weight.data()
prenet.weight.data()

# %%

plot_decision_boundary(lambda x: nd.argmax(net(nd.array(x)), axis = 1).asnumpy())

# %%

X, y = sklearn.datasets.make_circles(200, factor=.3, noise=.05)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# %%

tree = Tree(droprate_init = 0.01)

net = gluon.nn.Sequential()

with net.name_scope():
  prenet = gluon.nn.Dense(5, activation = "tanh")
  net.add(prenet)
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

prenet.weight.data()

# %%

prenet.weight.data()