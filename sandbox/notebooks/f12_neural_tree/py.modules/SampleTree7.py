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
    h = 0.5
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# %%

X, y = sklearn.datasets.make_circles(n_samples=200, factor=.3, noise=.05)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# %%

tree = Tree(droprate_init = 0.01)

net = gluon.nn.Sequential()

with net.name_scope():
  prenet = gluon.nn.Dense(3, activation = "tanh")
  net.add(prenet)
  net.add(tree)
  net.add(gluon.nn.Dense(2))

net.collect_params().initialize(mx.init.Normal(sigma = 0.1), force_reinit = True, ctx = model_ctx)
error = gluon.loss.SoftmaxCrossEntropyLoss()

# %%

X, y = shuffle(X, y)

for data, target in zip(np.split(X, 20), np.split(y, 20)):

  data = nd.array(data).as_in_context(model_ctx)
  target = nd.array(target).as_in_context(model_ctx)

  tree._grow(prenet(data))

  less = net.collect_params()
  for key in list(less._params.keys()):
    if less[key].shape is None:
      less._params.pop(key)

  trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 0.1})

  with mx.autograd.record():
    loss = error(net(data), target)

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
      if (value == hit_value):
        tree._prune(node)

  print(len(tree._routerlayer))

# %%

tree._grow(prenet(data))

less = net.collect_params()
for key in list(less._params.keys()):
  if less[key].shape is None:
    less._params.pop(key)

trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 0.1})

with mx.autograd.record():
  loss = error(net(data), target)

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
    if (value == hit_value):
      tree._prune(node)

print(len(tree._routerlayer))

# %%

prenet(data)

# %%

plot_decision_boundary(lambda x: nd.argmax(net(nd.array(x)), axis = 1).asnumpy())