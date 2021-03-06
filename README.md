# response-logic
[A network inference method based on a simple response logic with minimal presumptions](https://doi.org/10.1093/bioinformatics/btz326). This is the Python package. Various response logic projects can be found in a dedicated [repository](https://github.com/GrossTor/response-logic-projects).

This Python package works with clingo version 5.5.0 which is part of [Potassco, the Potsdam Answer Set Solving Collection](https://potassco.org/).
It can be [installed](https://github.com/potassco/clingo/blob/master/INSTALL.md) for example with [Anaconda](https://www.anaconda.com/):

```
conda install -c potassco clingo=5.5.0
```

Then you can install the response-logic package with [pip](https://pypi.org/project/pip/):

```
pip install git+https://github.com/GrossTor/response-logic#egg=response_logic
```

The response-logic package was used in various reverse engineering projects, which can be found in the [response-logic-projects repository](https://github.com/GrossTor/response-logic-projects). This includes the `toy_model` project that explains how to use the response-logic package.

The response logic approach can be cited with the following publication: [Robust network inference using response logic](https://doi.org/10.1093/bioinformatics/btz326).
