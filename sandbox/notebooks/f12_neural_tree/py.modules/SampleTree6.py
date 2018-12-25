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
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# %%

# X, y = sklearn.datasets.make_circles(n_samples=400, factor=.3, noise=.05)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# %%

net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Dense(10, activation = "tanh"))
    net.add(gluon.nn.Dense(2))

net.collect_params().initialize(mx.init.Normal(sigma = 0.1), ctx = model_ctx)
error = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# %%

for e in range(1000):
  X, y = shuffle(X, y)
  for data, target in zip(np.split(X, 20), np.split(y, 20)):
    data = nd.array(data).as_in_context(model_ctx)
    target = nd.array(target).as_in_context(model_ctx)
    with mx.autograd.record():
      output = net(data)
      loss = error(output, target)
    loss.backward()
    trainer.step(data.shape[0])

# %%

# Apparently, if you don't have as_in_context, it doesn't work

# for e in range(1000):
#   X, y = shuffle(X, y)
#   for data, target in zip(np.split(X, 20), np.split(y, 20)):
#     with mx.autograd.record():
#       output = net(nd.array(data))
#       loss = error(output, nd.array(target))
#     loss.backward()
#     trainer.step(data.shape[0])

# %%

nd.argmax(net(nd.array(data)), axis=1)
target

# %%

plot_decision_boundary(lambda x: nd.argmax(net(nd.array(x)), axis = 1).asnumpy())
