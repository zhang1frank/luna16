# %%

import torch
from torch.nn.parameter import Parameter
from torch import functional as F
from torch.nn import init
from torch.nn import Module

# %%

class Node(Module):

  def __init__(self, center, radius=torch.tensor(0.), decision=None, left=None, right=None):
    super(Node, self).__init__()
    self.center = Parameter(center, requires_grad=False)
    self.radius = Parameter(radius, requires_grad=False)
    self.children = {'decision': decision, 'left': left, 'right': right}

  def call(self, inputs):
    pass

# %%

class Decision(Module):
  def __init__(self, weight, bias, sharpness, tau, decision, side):
    super(Decision, self).__init__()
    self.sharpness = Parameter(sharpness, requires_grad=True)
    self.tau = Parameter(sharpness, requires_grad=False)
    self.split = 

    self.kernel_initializer = tf.constant_initializer(weight)
    self.bias_initializer = tf.constant_initializer(bias)

    self.split = tf.keras.layers.Dense(
      units=1,
      kernel_initializer=self.kernel_initializer,
      bias_initializer=self.bias_initializer,
      trainable=True
    )
    self.parent = {'decision': decision, 'side': side}

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

Parameter(torch.tensor(1.))