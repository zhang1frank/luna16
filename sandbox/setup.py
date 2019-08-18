from setuptools import setup

setup(
  name = "luna16.sandbox.python",
  version ="0.1.0",
  install_requires = [
    "matplotlib >= 2.2",
    "mxnet >= 1.2",
    # "tensorflow >= 2.0.0-beta1"
    "radio @ git+https://github.com/analysiscenter/radio.git",
    "tqdm >= 4.32",
    # "torch >= 1.1"
  ],
  dependency_links = []
)
