# %%
import tensorflow as tf

# %%

class Node(tf.keras.layers.Layer):
  def __init__(self, center, radius=0, decision=None, left=None, right=None):
    super(Node, self).__init__()
    self.center_initializer = tf.constant_initializer(center)
    self.radius_initializer = tf.constant_initializer(radius)
    self.children = {decision: 0, left: -1, right: 1}

  def build(self, input_shape):
    self.center = self.add_weight(
      'center',
      shape=input_shape[-1],
      initializer=self.center_initializer,
      trainable=False
    )
    self.radius = self.add_weight(
      'radius',
      shape=1,
      initializer=self.radius_initializer,
      trainable=False
    )

  def call(self, inputs):
    pass

# %%

class Decision(tf.keras.layers.Layer):
  def __init__(self, weight, bias, sharpness, tau, decision, side):
    super(Decision, self).__init__()
    self.kernel_initializer = tf.constant_initializer(weight)
    self.bias_initializer = tf.constant_initializer(bias)
    self.sharpness_initializer = tf.constant_initializer(sharpness)
    self.tau_initializer = tf.constant_initializer(tau)
    self.split = tf.keras.layers.Dense(
      units=1,
      kernel_initializer=self.kernel_initializer,
      bias_initializer=self.bias_initializer,
      trainable=True
    )
    self.parent = {decision: side}

  def build(self, input_shape):
    self.sharpness = self.add_weight(
      'sharpness',
      shape=1,
      initializer=self.sharpness_initializer,
      trainable=True
    )
    self.tau = self.add_weight(
      'tau',
      shape=1,
      initializer=self.tau_initializer,
      trainable=False
    )

  def call(self, inputs):
    x = self.split(inputs)
    a = tf.math.softplus(self.sharpness)
    return tf.math.sigmoid(a * x)

# %%

class Rule(tf.keras.layers.Layer):
  def __init__(self, layer, decision, side):
    super(Rule, self).__init__()
    self.layer = layer
    self.parent = {decision: side}

  def build(self, input_shape):
    pass

  def call(self, inputs):
    return self.layer(inputs)

# %%

class Tree(tf.keras.Model):
  def __init__(self):
    super(Tree, self).__init__()

  def build(self, input_shape):
    a = Decision([1.], [0.], [1.], [0.], None, None)
    b = Decision([1.], [2.], [1.], [0.], a, 1)
    c = Decision([1.], [1.], [1.], [0.], b, 0)
    d = Decision([1.], [3.], [1.], [0.], b, 1)
    w = Rule(tf.keras.layers.Dense(2), a, 0)
    v = Rule(tf.keras.layers.Dense(2), c, 0)
    y = Rule(tf.keras.layers.Dense(2), d, 0)
    z = Rule(tf.keras.layers.Dense(2), d, 1)
    x = Rule(tf.keras.layers.Dense(2), c, 1)
    self.blocks = []
    self.splits = [a, b, c, d]
    self.leaves = [w, v, x, y, z]

  def call(self, inputs):
    # x = tf.stack([l(inputs) for l in self.leaves])
    # x = tf.stack([s(inputs) for s in self.splits])
    leaf_output = []
    leaf_weight = []
    for leaf in self.leaves:
      leaf_path = []
      split = leaf.parent
      while 1:
        side = next(iter(split.values()))
        decision = next(iter(split.keys()))
        leaf_path.append(
          tf.math.abs(0**side - decision(inputs))
        )
        split = decision.parent
        if None in split: break
      leaf_weight.append(tf.math.reduce_prod(leaf_path, axis=0))
      leaf_output.append(leaf(inputs))

    return tf.reduce_sum(tf.stack(leaf_weight) * tf.stack(leaf_output), axis=0)

# %%

def extend_tree(tree, inputs, lambda):
  def sample_block(
    parent,
    inputs,
    parent_previous_center = None,
    parent_previous_radius = None
  ):
    pass

  def extend_block(node, inputs):
    n = 1
    N = inputs.shape[0]
    data_center = tf.math.reduce_mean(inputs, axis=0)
    data_variance = tf.math.reduce_variance(inputs, axis=0)
    new_center = (n*node.center + N*data_center) / (n + N)
    new_variance = ((n * (node.center + (0.5 * node.radius) ** 2) + N * (data_center + data_variance)) / (n + N)) - new_center
    new_radius = 2 * (tf.math.reduce_max(new_variance) ** 0.5)

    pass

  extend_block(tree.blocks[0], inputs)

# %%

tf.math.reduce_std(tf.stack([[1., 50., 20.], [0., 0., 40.]]), axis=0)

tf.math.reduce_mean(tf.stack([[1., 50., 20.], [0., 0., 40.]]), axis=0)

tf.stack([[1., 50., 0.], [0., 0., 0.]]) + 1

tf.distributions.Exponential(0.5)

# %%

a = Node([5, 6, 1])
b = Decision([5., 6.], [1.], [2.], [5.], None, None)

b(tf.zeros((2, 2)))

b.weights

c = Tree()
c(tf.stack([[1.], [2.]]))

c.get_weights()
next(iter(c.layers[-1].parent.values()))

tf.stack([[1.], [1.]])

a = 5

tf.stack([[1], [1]])

n = 1
N = 2


# %%

class StochasticNetworkDepth(tf.keras.Sequential):
  def __init__(self, layers, pfirst=1.0, plast=0.5,**kwargs):
    self.pfirst = pfirst
    self.plast = plast
    super(StochasticNetworkDepth, self).__init__(layers,**kwargs)

  def build(self, input_shape):
    self.depth = len(self.layers)
    self.plims = np.linspace(self.pfirst, self.plast, self.depth + 1)[:-1]
    super(StochasticNetworkDepth, self).build(input_shape.as_list())

  @autograph.convert(optional_features=autograph.Feature.ALL)
  def call(self, inputs):
    training = tf.cast(K.learning_phase(), dtype=bool)
    if not training:
      count = self.depth
      return super(StochasticNetworkDepth, self).call(inputs), count

    p = tf.random_uniform((self.depth,))

    keeps = (p <= self.plims)
    x = inputs

    count = tf.reduce_sum(tf.cast(keeps, tf.int32))
    for i in range(self.depth):
      if keeps[i]:
        x = self.layers[i](x)

    # return both the final-layer output and the number of layers executed.
    return x, count