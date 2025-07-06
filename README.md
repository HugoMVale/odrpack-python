# odrpack (-python)

[![Test-Linux](https://github.com/HugoMVale/odrpack-python/actions/workflows/test-linux.yml/badge.svg)](https://github.com/HugoMVale/odrpack-python/actions)
[![codecov](https://codecov.io/gh/HugoMVale/odrpack-python/graph/badge.svg?token=B9sFyJiweC)](https://codecov.io/gh/HugoMVale/odrpack-python)
[![Latest Commit](https://img.shields.io/github/last-commit/HugoMVale/odrpack-python)](https://img.shields.io/github/last-commit/HugoMVale/odrpack-python)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/HugoMVale/odrpack-python)

## Description

This Python package provides bindings for the well-known weighted orthogonal distance regression
(ODR) solver [odrpack95]. This design ensures that users benefit from the performance and reliability
of the original Fortran implementation, while working within the modern Python ecosystem.  

ODR, also known as [errors-in-variables regression], is designed primarily for instances when both
the explanatory and response variables have significant errors. 

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Total_least_squares.svg/220px-Total_least_squares.svg.png" width="250" alt="Deming regression; special case of ODR." style="margin-right: 10px;">
  <img src="https://github.com/HugoMVale/odrpack-python/blob/main/docs/examples/ellipse.png" width="400" alt="Estimated ellipse parameters.">
</p>

[errors-in-variables regression]: https://en.wikipedia.org/wiki/Errors-in-variables_models
[odrpack95]: https://github.com/HugoMVale/odrpack95


## Installation

You can install the package via pip:

```sh
pip install odrpack
```

## Documentation and Usage

The following example demonstrates a simple use of the package. For more comprehensive examples and explanations, please refer to the [documentation](https://hugomvale.github.io/odrpack-python/) pages.

```py
from odrpack import odr_fit
import numpy as np

xdata = np.array([0.982, 1.998, 4.978, 6.01])
ydata = np.array([2.7, 7.4, 148.0, 403.0])

beta0 = np.array([2.0, 0.5])
bounds = (np.array([0.0, 0.0]), np.array([10.0, 0.9]))

def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    "Model function."
    return beta[0] * np.exp(beta[1]*x)

sol = odr_fit(f, xdata, ydata, beta0, bounds=bounds)

print("beta:", sol.beta)
print("delta:", sol.delta)
```

```sh
beta: [1.63336897 0.9       ]
delta: [-0.36885696 -0.31272648  0.02929022  0.11031872]
```
