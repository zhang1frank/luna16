from setuptools import setup

setup(
  name = "luna16.analysis.inference.python",
  version ="0.1.0",
  install_requires = [
    "matplotlib >= 2.2",
    "radio == 0.1.0"
  ],
  dependency_links = [
    "git+https://github.com/analysiscenter/radio.git#egg=radio-0.1.0"
  ]
)
