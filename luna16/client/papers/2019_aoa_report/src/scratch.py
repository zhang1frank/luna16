def add_node(x, leaf):
  N = x.shape[0]
  x_mean = nd.mean(x, axis=0)
  x_var = (N ** -1) * nd.sum((x - x_mean) ** 2, axis=0)
  x_radius = 2 * (nd.max(x_var) ** 0.5)
  node = Node(x_mean, x_radius)
  tree.nodes.add(node)
  if leaf.parent['node'] is not None:
    if leaf.parent['side'] == 0:
      print('yes')
      leaf.parent['node'].child['left'] = node
    if leaf.parent['side'] == 1:
      print('ha')
      leaf.parent['node'].child['right'] = node

  leaf.parent['node'] = node
  return node

def add_split(x, leaf, p_tau):
  center = leaf.parent['node'].center.data()
  radius = leaf.parent['node'].radius.data()
  tau = p_tau + nd.random.exponential(radius ** -1)
  if (len(x) < 2): return
  while 1:
    s = nd.random.normal(shape=(2, x.shape[-1]))
    s = s / nd.norm(s, axis=-1, keepdims=True)
    r = nd.random.uniform(low=nd.array([0]), high=radius)
    r = r * nd.random.uniform() ** (1/3)
    if nd.sign(s[0][-1]) > 0:
      weight = s[0]
      bias = nd.dot(s[0], -1 * r * (s[1] + center))
      y = nd.sign(nd.dot(data, weight) + bias)
      if nd.abs(nd.sum(y)) != len(y): break

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
  return split

def add_leaf(leaf):
  leaf.parent['side'] = 0
  new_leaf = Leaf(
    layer=tree.layer_initializer(),
    node=leaf.parent['node'],
    decision=leaf.parent['decision'],
    side=1
  )
  tree.leaves.add(new_leaf)
  return new_leaf

def tesselate(x, leaf, p_tau):
  split = add_split(x, leaf, p_tau)
  if split is None: return
  side = nd.sign(split.split(x))
  order = nd.argsort(side, axis = None)
  x = x[order, :]
  side = side[order, :]
  orderside = nd.argsort(side, axis=0) * side
  cutpt = nd.argsort(orderside, axis=None, dtype='int32')[0].asscalar() + 1
  x_l = x[0:cutpt]
  x_r = x[cutpt:None]
  node_l = add_node(x_l, leaf)
  new_leaf = add_leaf(leaf)
  node_r = add_node(x_r, new_leaf)
  tesselate(x_l, leaf, split.tau.data())
  tesselate(x_r, new_leaf, split.tau.data())

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

# %%

def extend_tree(x, tree):

  def update_centroid(x, mean=None, var=None):
    n = 1
    N = x.shape[0]
    if N == 0: return mean, var
    x_mean = nd.mean(x, axis=0)
    x_var = (N ** -1) * nd.sum((x - x_mean) ** 2, axis=0)
    if mean is None or var is None: return x_mean, nd.max(x_var)
    z_mean = (n * mean + N * x_mean) / (n + N)
    z_var = ((n * (mean + var) + N * (x_mean + x_var)) / (n + N)) - z_mean
    return z_mean, nd.max(z_var)

  def add_node(x, leaf, p_mean=None, p_var=None):
    z_mean, z_var = update_centroid(x, p_mean, p_var)
    z_radius = 2 * (z_var ** 0.5)
    node = Node(z_mean, z_radius)
    tree.nodes.add(node)
    if leaf.parent['node'] is not None:
      if leaf.parent['side'] == 0: leaf.parent['node'].child['left'] = node
      if leaf.parent['side'] == 1: leaf.parent['node'].child['right'] = node

    leaf.parent['node'] = node
    return node

  def add_leaf(x, leaf):
    leaf.parent['side'] = 0
    new_leaf = Leaf(
      layer=tree.layer_initializer(),
      node=leaf.parent['node'],
      decision=leaf.parent['decision'],
      side=1
    )
    tree.leaves.add(new_leaf)
    return new_leaf

  def add_split(x, leaf, p_tau, p_mean=None):
    center = leaf.parent['node'].center.data()
    radius = leaf.parent['node'].radius.data()
    tau = p_tau + nd.random.exponential(radius ** -1)
    if (radius == 0): return
    shape = x.shape[-1] if x.shape != 0 else p_mean.shape
    while 1:
      s = nd.random.normal(shape=(2, x.shape[-1]))
      s = s / nd.norm(s, axis=-1, keepdims=True)
      r = nd.random.uniform(low=nd.array([0]), high=radius)
      r = r * nd.random.uniform() ** (1/3)
      # have to multiply all future input by the sign of the last dimension
      # this preserves the proper left-right orientation of normal trees
      # except this might mess up the gradients going to the inputs
      # so might need to just accept that left, right just means which side is +
      # or could just alter weights after fact to ensure last weight is +
      # this means just swapping the signs of weight and bias as necessary
      # this doesn't mess up probs just as long as left is - and right is +
      # but - and + might not correspond to left and right in real space
      if nd.sign(s[0][-1]) > 0:
        weight = s[0]
        bias = nd.dot(s[0], -1 * r * (s[1] + center))
        y = nd.sign(nd.dot(data, weight) + bias)
        if p_mean is not None:
          y = nd.concat(y, nd.sign(nd.dot(p_mean, weight) + bias), dim=0)

        if nd.abs(nd.sum(y)) != len(y): break

    split = Split(
      weight=weight,
      bias=bias,
      sharpness=3 / radius,
      tau=tau,
      decision=leaf.parent['decision'],
      side=leaf.parent['side']
    )
    tree.splits.add(split)
    leaf.parent['node'].child['split'] = split
    leaf.parent['decision'] = split
    return split

  def recurse(x, node, p_tau, p_mean=None, p_var=None):
    n_mean = node.center.data()
    n_var = (0.5 * node.radius.data()) ** 2
    z_mean, z_var = update_centroid(x, n_mean, n_var)
    z_radius = 2 * (z_var ** 0.5)
    node.center.set_data(z_mean)
    node.radius.set_data(z_radius)
    if node.child['decision'] is not None:
      E = nd.random.exponential(z_radius ** -1)
      node.child['decision'].tau.set_data(p_tau + E)
      split = node.child['decision']
    else:
      leaf = next(l for l in tree.leaves if l.parent['node'] == node)
      split = add_split(x, leaf, p_tau, p_mean)

    if split is None: return
    side = nd.sign(split.split(x))
    order = nd.argsort(side, axis = None)
    x = x[order, :]
    side = side[order, :]
    if side[0] > 0:
      x_l = nd.array([])
      x_r = x
    elif side[-1] < 0:
      x_l = x
      x_r = nd.array([])
    else:
      orderside = nd.argsort(side, axis=0) * side
      cutpt = nd.argsort(orderside, axis=None, dtype='int32')[0].asscalar() + 1
      x_l = x[0:cutpt]
      x_r = x[cutpt:None]

    p_side = nd.sign(split.split(nd.stack(p_mean)))[0] if p_mean is not None else 0
    if node.child['decision'] is not None:
      node_l = node.child['left']
      node_r = node.child['right']
    else:
      new_leaf = add_leaf(x, leaf)
      node_l = add_node(
        x_l,
        leaf,
        p_mean if p_side < 0 else None,
        p_var if p_side < 0 else None
      )
      node_r = add_node(
        x_r,
        new_leaf,
        p_mean if p_side > 0 else None,
        p_var if p_side > 0 else None
      )

    n_side = nd.sign(split.split(nd.stack(n_mean)))[0]
    recurse(
      x_l,
      node_l,
      split.tau.data(),
      n_mean if n_side < 0 else None,
      n_var if n_side < 0 else None
    )
    recurse(
      x_r,
      node_r,
      split.tau.data(),
      n_mean if n_side > 0 else None,
      n_var if n_side > 0 else None
    )


  with tree.name_scope():
    if len(tree.nodes) == 0:
      leaf = tree.leaves[0]
      node = add_node(x, leaf)
      recurse(x, node, 0)
    else:
      recurse(x, tree.nodes[0], 0)

# %%

def extend_tree(x, tree, lambda):

  def update_centroid(x, mean=None, var=None):
    n = 1
    N = x.shape[0]
    if N == 0:
      return mean, var
    x_mean = nd.mean(x, axis=0)
    x_var = (N ** -1) * nd.sum((x - x_mean) ** 2, axis=0)
    if mean is None or var is None:
      return x_mean, nd.max(x_var)
    z_mean = (n * mean + N * x_mean) / (n + N)
    z_var = ((n * (mean + var) + N * (x_mean + x_var)) / (n + N)) - z_mean
    return z_mean, nd.max(z_var)

  def update_block(x, node, p_tau):
    n_mean = node.center.data()
    n_var = (0.5 * node.radius.data()) ** 2
    z_mean, z_var = update_centroid(x, n_mean, n_var)
    z_radius = 2 * (nd.max(z_var) ** 0.5)
    node.center.set_data(z_mean)
    node.radius.set_data(z_radius)
    if node.child['decision'] is not None:
      E = nd.random.exponential(z_radius ** -1)
      node.child['decision'].tau.set_data(p_tau + E)
      x_l, x_r = split_data(x, node.child['decision'])
      update_block(x_l, node.child['left'], node.child['decision'].tau.data())
      update_block(x_r, node.child['right'], node.child['decision'].tau.data())
    else:
      sample_split(x, leaf)

  def sample_block(x, p_mean=None, p_var=None):
    z_mean, z_var = update_centroid(x, p_mean, p_var)
    z_radius = 2 * (nd.max(z_var) ** 0.5)
    node = Node(z_mean, z_radius)
    tree.nodes.add(node)
    return node

  def sample_split(x, leaf, ptau):
    radius = leaf.parent['node'].radius.data()
    tau = ptau + nd.random.exponential(radius ** -1)
    if (tau > lambda) or (radius == 0):
      return

    split = Split(tau, leaf.parent['decision'], leaf.parent['side'])
    tree.splits.add(split)

    x_l, x_r = split_data(x, split)
    node_l = sample_block(x_l)
    node_r = sample_block(x_r)

    leaf.parent['node'].child['decision'] = split
    leaf.parent['node'].child['left'] = node_l
    leaf.parent['node'].child['right'] = node_r

    new_leaf = clone_leaf(leaf)
    tree.leaves.add(new_leaf)

    new_leaf.parent['node'] = node_l
    new_leaf.parent['decision'] = split
    new_leaf.parent['side'] = 0
    leaf.parent['node'] = node_r
    leaf.parent['decision'] = split
    leaf.parent['side'] = 1

    sample_split(x_l, new_leaf)
    sample_split(x_r, leaf)

  with tree.name_scope():
    if len(tree.nodes) > 0:
      update_block(x, tree.nodes[0], 0)
    else:
      leaf = tree.leaves[0]
      node = sample_block(x)
      leaf.parent['node'] = node
      sample_split(x, leaf)

# %%

def extend_tree(tree, x, lambda):

  def merge_centroids(x, mean=None, var=None):
    n = 1
    N = x.shape[0]
    if N == 0:
      return mean, var

    x_mean = nd.mean(x, axis=0)
    x_var = (N ** -1) * nd.sum((x - x_mean) ** 2, axis=0)

    if mean is None or var is None:
      return x_mean, x_var

    z_mean = (n * mean + N * x_mean) / (n + N)
    z_var = ((n * (mean + var) + N * (x_mean + x_var)) / (n + N)) - z_mean

    return z_mean, z_var

  def sample_block(parent, x, side, node, p_mean = None, p_var = None):
    n_mean, n_var = None
    z_mean, z_var = merge_centroids(x, p_mean, p_var)
    z_radius = 2 * (nd.max(z_var) ** 0.5)
    if node is None:
      node = Node(z_mean, z_radius)
      tree.nodes.add(node)
      if parent is not None:
        if side == 0:
          parent.child['left'] = node
        if side == 1:
          parent.child['right'] = node
    else:
      n_mean = node.center.data()
      n_var = (0.5 * node.radius.data()) ** 2
      node.center.set_data(z_mean)
      node.radius.set_data(z_radius)

    if parent is None:
      ptau = 0
    else:
      ptau = parent.child['decision'].tau

    E = nd.random.exponential(z_radius ** -1)
    tau = p_tau + E
    if node.child['decision'] is not None:
      node.child['decision'].tau.set_data(tau)
    elif (tau < lambda) and (z_radius > 0):
      split = Split(tau, parent.child['decision'], side)
      node.child['decision'] = split
      tree.splits.add(node)
    else:
      return

    sample_block(node, x_l, 0, node.child['left'], n_mean, n_var)
    sample_block(node, x_r, 1, node.child['right'], n_mean, n_var)

  with tree.name_scope():
    if len(tree.nodes) > 0:
      sample_block(None, x, None, tree.nodes[0])
    else:
      sample_block(None, x, None)

# %%

nd.sum((nd.array([[1, 2, 0], [2, 3, 5]]) - nd.mean(nd.array([[1, 2, 0], [2, 3, 5]]), axis=0)) ** 2, axis=0)
nd.mean(nd.array([[1, 2, 0]]), axis=0)

nd.random.exponential(nd.array([.01]))

nd.mean(nd.array([]), axis=0)
nd.array([]).shape

n = 1
N = 2

n, N

# %%

nd.concat(*[nd.array([[1.]]), nd.array([[2]])])

nd.array([[1, 2], [2, 2], [3, 2]]).shape

a = Tree()

len(a.nodes)

a.collect_params().initialize()

a(nd.array([[1, 2], [2, 2], [3, 2]]))

nd.stack(*b[0], axis=-1)
nd.stack(*b[1], axis=-1)

nd.stack(*b[0],axis=-1) * nd.stack(*b[1],axis=-1)
nd.sum(nd.stack(*b[0],axis=-1) * nd.stack(*b[1],axis=-1), axis=-1)

for leaf in a.leaves:
  split = leaf.parent
  while 1:
    print(split)
    if split['decision'] is None: break
    split = split['decision'].parent

# %%

nd.exp(nd.array([1]))

Node(nd.array([0, 1]), nd.array([0])).params
a = Split(nd.array([[0, 0]]), nd.array([0]), nd.array([1]), nd.array([0]), None, None)

a.collect_params()
a.params.get('tau')

nd.array([0, 0]).shape[-1]
a(nd.array([[2, 2], [1, 1]]))

nd.array([[1,1]]).shape

a = nn.Dense(1)
a.initialize()
a(nd.array([[2,1],[3,1]]))
[y for x in a.params]
a.params

nd.array([1,2,3])