# response-logic
A network inference method based on a simple response logic with minimal presumptions. This is the Python package. Various response logic projects can be found in a dedicated [repository](https://github.com/GrossTor/response-logic-projects).

This Python 3 package requires clingo version 5.2.2 which is part of [Potassco, the Potsdam Answer Set Solving Collection](https://potassco.org/).
It can be [installed](https://github.com/potassco/clingo/blob/master/INSTALL.md) for example with [Anaconda](https://www.anaconda.com/):

```
conda install -c potassco clingo=5.2.2
```

Then you can install the response-logic package with [pip](https://pypi.org/project/pip/):

```
pip install git+https://github.com/GrossTor/response-logic#egg=response_logic
```

The response-logic package was used in various reverse engineering projects, which can be found in the [response-logic-projects repository](https://github.com/GrossTor/response-logic-projects). This includes the `toy_model` project that explains how to use the response-logic package.
