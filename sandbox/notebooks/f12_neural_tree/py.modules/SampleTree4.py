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
X = np.sort(15 * rng.rand(80, 1), axis = 0)
X_test = np.arange(0.0, 15.0, 0.01)[:, np.newaxis]
y = np.sin(X).ravel()

# %%

tree = Tree(droprate_init = 0.01)
tree.collect_params().initialize(force_reinit = True)
error = gluon.loss.L2Loss()

# %%

X, y = shuffle(X, y)

data = X[:10]
target = y[:10]

# %%

tree._grow(nd.array(data))

# %%



# %%

X, y = shuffle(X, y)

# data = X[:5]
# target = y[:5]

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

len(tree._routerlayer)

set([decision._gate().asscalar() for decision in tree._routerlayer._children.values()])

next(iter(tree._structure.items()))[0]._decision._gate()

max(set([x for x in after if x is not None]), key = after.count)

mode = max(set([x for x in after if x is not None]), key = after.count)
after.count(mode)
hitlist = mode if after.count(mode) > 1 else None

for node, value in zip(list(tree._embeddlayer._children.values()), after):
  print(value)

for node, value in zip(list(tree._embeddlayer._children.values()), after):
  if (value == hitlist):
    tree._prune(node)

# %%

root = next(iter(tree._structure.items()))[0]
router_d, router_mat_d, weight_d, embedd_d = tree._contextify(nd.array([[1.75]]))(root)

router = nd.stack(*[router_d[key] for key in sorted(router_d)], axis = -1)
weight = nd.stack(*[weight_d[key] for key in sorted(weight_d)], axis = -1)

embedd = nd.stack(*[embedd_d[key] for key in sorted(embedd_d)], axis = 0)
router_mat = nd.stack(*[router_mat_d[key] for key in sorted(router_mat_d)], axis = 1)

where = nd.argmin(nd.abs(router + 0.5), axis = 1)

head = nd.concat(*[router_mat[i][k] for i, k in enumerate(where)], dim = 0)

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

plt.figure()
plt.scatter(X, y, s = 20, edgecolor = "black", c = "darkorange", label = "data")
y_1 = tree(nd.array(X_test)).asnumpy()
plt.plot(X_test, y_1, color = "cornflowerblue", linewidth = 2)

plt.show()

# %%
