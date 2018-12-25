# %%

import matplotlib.pyplot as plt

from Tree import Tree

import mxnet as mx
import numpy as np
from mxnet import gluon, nd

from sklearn.utils import shuffle

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
data_ctx = ctx
model_ctx = ctx

# %%

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis = 0)
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y = np.sin(X).ravel()

# %%

tree = Tree(droprate_init = 0.01)

net = gluon.nn.Sequential()

with net.name_scope():
  net.add(tree)
  net.add(gluon.nn.Dense(1))

error = gluon.loss.L2Loss()
net.collect_params().initialize(mx.init.Normal(sigma = 0.1), force_reinit = True)

# %%

X, y = shuffle(X, y)

for data, target in zip(np.split(X, 10), np.split(y, 10)):

  tree._grow(nd.array(data))

  less = net.collect_params()
  for key in list(less._params.keys()):
    if less[key].shape is None:
      less._params.pop(key)

  trainer = gluon.Trainer(less, 'sgd', {'learning_rate': 0.5})

  with mx.autograd.record():
    loss = error(net(nd.array(data)), nd.array(target))

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

plt.figure()
plt.scatter(X, y, s = 20, edgecolor = "black", c = "darkorange", label = "data")
y_1 = net(nd.array(X_test)).asnumpy()
plt.plot(X_test, y_1, color = "cornflowerblue", linewidth = 2)

plt.show()