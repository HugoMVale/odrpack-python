# odrpack

[![Test-Linux](https://github.com/HugoMVale/odrpack-python/actions/workflows/test-linux.yml/badge.svg)](https://github.com/HugoMVale/odrpack-python/actions)
[![codecov](https://codecov.io/gh/HugoMVale/odrpack-python/graph/badge.svg?token=B9sFyJiweC)](https://codecov.io/gh/HugoMVale/odrpack-python)
[![Latest Commit](https://img.shields.io/github/last-commit/HugoMVale/odrpack-python)](https://img.shields.io/github/last-commit/HugoMVale/odrpack-python)

## Description

`odrpack` is a package for weighted orthogonal distance regression (ODR), also known as [errors-in-variables regression]. 
It is designed primarily for instances when both the explanatory and response variables have significant errors. 
The package implements a highly efficient algorithm for minimizing the sum of the squares of the weighted orthogonal
distances between each data point and the curve described by the model equation, subject to parameter bounds. The nonlinear
model can be either explicit or implicit. Additionally, `odrpack` can be used to solve the ordinary least squares problem where all of
the errors are attributed to the observations of the dependent variable.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/Total_least_squares.svg/220px-Total_least_squares.svg.png" width="200" alt="Deming regression; special case of ODR.">
</p>

[errors-in-variables regression]: https://en.wikipedia.org/wiki/Errors-in-variables_models

## Python Bindings for Fortran Implementation

The `odrpack` Python package provides convient bindings for the efficient ODR solver available
in the Fortran [odrpack95] repository. This design ensures that users benefit from the performance
and reliability of the original Fortran implementation, while working within the modern Python
ecosystem.  

[odrpack95]: https://github.com/HugoMVale/odrpack95


## Installation

You can install the package via pip:

```bash
pip install odrpack
```

## Usage

```py
from odrpack import odr
import numpy as np

beta0 = np.array([2., 0.5])
lower = np.array([0., 0.])
upper = np.array([10., 0.9])
x = np.array([0.982, 1.998, 4.978, 6.01])
y = np.array([2.7, 7.4, 148.0, 403.0])


def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    "Model function."
    return beta[0] * np.exp(beta[1]*x)


def fjacb(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    "Model jacobian w.r.t. the model parameters."
    jac = np.zeros((beta.size, x.size))
    jac[0, :] = np.exp(beta[1]*x)
    jac[1, :] = beta[0]*x*np.exp(beta[1]*x)
    return jac


def fjacd(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    "Model jacobian w.r.t. the error in the explanatory variable."
    return beta[0] * beta[1] * np.exp(beta[1]*x)


sol = odr(f, beta0, y, x, lower=lower, upper=upper,
          fjacb=fjacb, fjacd=fjacd,
          job=20, iprint=1001)

print("\n beta:", sol.beta)
print("\n delta:", sol.delta)
```