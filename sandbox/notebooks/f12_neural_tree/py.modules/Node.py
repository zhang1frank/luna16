import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import Block

class Custom_Block(Block):
  def __init__(self, **kwargs):
    super(Custom_Block, self).__init__(**kwargs)

  def _init_param(self, param_name, value):
    param =  self.params.get(param_name)
    if (value is not None and param.shape is None):
      param.set_data(value)
      param.initialize(init = mx.init.Constant(value))
    elif (value is not None):
      param.set_data(value)


class Gate(Custom_Block):
  def __init__(self, droprate_init, temperature, limit_lo, limit_hi, **kwargs):
    super(Gate, self).__init__(**kwargs)
    self._temperature = temperature
    self._limit_lo = limit_lo
    self._limit_hi = limit_hi

    droprate_init = nd.array([droprate_init])
    qz_loga = nd.log(1 - droprate_init) - nd.log(droprate_init)

    with self.name_scope():
      self._qz_loga = self.params.get(
        "qz_loga", init = mx.init.Constant(qz_loga),
        allow_deferred_init = True)
      self._init_param("qz_loga", qz_loga)

  def forward(self, x = 0):
    if (mx.autograd.is_training()):
      u = nd.random.uniform(0, 1)
      s = nd.log(u) - nd.log(1 - u) + self._qz_loga.data()
      if (self._temperature == 0):
        s = nd.sign(s)
      else:
        s = nd.sigmoid(s / self._temperature)

    else:
      s = nd.sigmoid(self._qz_loga.data())

    s = s * (self._limit_hi - self._limit_lo) + self._limit_lo

    return nd.minimum(1, nd.maximum(s, 0))


class Box(Custom_Block):
  def __init__(self,
    parent = None, min_list = None, max_list = None, tau = None, **kwargs
  ):
    super(Box, self).__init__(**kwargs)
    with self.name_scope():
      self._parent = parent
      self._min_list = self.params.get(
        "min_list", grad_req = "null", init = mx.init.Constant(min_list),
        allow_deferred_init = True)
      self._max_list = self.params.get(
        "max_list", grad_req = "null", init = mx.init.Constant(max_list),
        allow_deferred_init = True)
      self._tau = self.params.get(
        "tau", grad_req = "null", init = mx.init.Constant(tau),
        allow_deferred_init = True)

      self._init_param("min_list", min_list)
      self._init_param("max_list", max_list)
      self._init_param("tau", tau)

  def forward(self, x):
    parent_tau = 0
    if (self._parent is not None):
      parent_tau = self._parent._box._tau.data()
    s = nd.sum(
      nd.maximum(x - self._max_list.data(), 0) +
      nd.maximum(self._min_list.data() - x, 0),
      axis = 1, keepdims = True
    )
    ans = nd.ones_like(s)
    if (self._tau.shape is not None):
      delta = self._tau.data() - parent_tau
      ans -= nd.exp(-1 * delta * s)

    return ans


class Decision(Custom_Block):
  def __init__(self, split, dim, gate, **kwargs):

    super(Decision, self).__init__(**kwargs)
    with self.name_scope():
      self._gate = gate()
      self._sharpness = self.params.get(
        "sharpness", init = mx.init.Constant(nd.array([1])),
        allow_deferred_init = True)
      self._split = self.params.get(
        "split", init = mx.init.Constant(split),
        allow_deferred_init = True)
      self._dim = self.params.get(
        "dim", grad_req = "null", init = mx.init.Constant(dim),
        allow_deferred_init = True)

      self._init_param("sharpness", nd.array([1]))
      self._init_param("split", split)
      self._init_param("dim", dim)

  def forward(self, x, crisp = False):
    pick_index = nd.broadcast_to(self._dim.data(), x.shape[0])
    x = nd.pick(x, pick_index, keepdims = True)
    x -= self._split.data()
    if (crisp == False):
      x *= nd.relu(self._sharpness.data()) * self._gate()

    return nd.tanh(x)


class Node(Custom_Block):
  def __init__(self,
    parent = None, min_list = None, max_list = None, tau = None,
    units = 1, weight_initializer = None, embedding = None,
    decision = None,
    **kwargs
  ):
    super(Node, self).__init__(**kwargs)
    with self.name_scope():
      self._box = Box(
        parent = parent, min_list = min_list, max_list = max_list, tau = tau
      )
      if (decision is not None):
        self._decision = decision()
      if (embedding is not None):
        self._embedding = self.params.get(
          "embedding", init = mx.init.Constant(embedding))
        self._init_param("embedding", embedding)
      else:
        self._embedding = self.params.get(
          "embedding", init = weight_initializer,
          shape = (units,)
        )
        self._embedding.initialize()
  def forward(self, x = 0):
    return self._embedding.data()
