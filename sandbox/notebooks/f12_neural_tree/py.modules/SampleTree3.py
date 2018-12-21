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
# y[::2] += 3 * (0.5 - rng.rand(40))

# %%

plt.figure()
plt.scatter(X, y, s = 20, edgecolor = "black", c = "darkorange", label = "data")

plt.show()

# %%

net = gluon.nn.Sequential()

with net.name_scope():
  forest = gluon.contrib.nn.Concurrent()
  with forest.name_scope():
    forest.add(*[Tree() for x in range(5)])
  net.add(forest)
  net.add(gluon.nn.Dense(1))

net.collect_params().initialize(mx.init.Normal(sigma = 0.1), force_reinit = True, ctx = model_ctx)
error = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})

# %%

net(nd.array(X))

tree = Tree()
tree._weightlayer

a = {0: 1}
1 in a

# %%

net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Dense(3, activation="tanh"))
    net.add(gluon.nn.Dense(3, activation="tanh"))
    net.add(gluon.nn.Dense(1))

net.collect_params().initialize(mx.init.Normal(sigma = 0.1), ctx = model_ctx)
error = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .05})

# %%

plt.figure()
plt.scatter(X, y, s = 20, edgecolor = "black", c = "darkorange", label = "data")
y_1 = net(nd.array(X_test)).asnumpy()
plt.plot(X_test, y_1, color = "cornflowerblue", linewidth = 2)

plt.show()

# %%

for data, target in zip(np.split(X, 20), np.split(y, 20)):
  print(data, target)

# %%

for e in range(100):
  X, y = shuffle(X, y)
  for data, target in zip(np.split(X, 10), np.split(y, 10)):
    with mx.autograd.record():
      output = net(nd.array(data))
      loss = error(output, nd.array(target))
    loss.backward()
    trainer.step(data.shape[0])

# %%

loss