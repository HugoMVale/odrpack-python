from pprint import pprint

import numpy as np

from odrpack import odr

beta0 = np.array([2., 0.5])
lower = np.array([0., 0.])
upper = np.array([10., 0.9])
x = np.array([0.982, 1.998, 4.978, 6.01])
y = np.array([2.7, 7.4, 148.0, 403.0])


def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] * np.exp(beta[1]*x)


def fjacb(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    jac = np.zeros((beta.size, x.size))
    jac[0, :] = np.exp(beta[1]*x)
    jac[1, :] = beta[0]*x*np.exp(beta[1]*x)
    return jac


def fjacd(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return beta[0] * beta[1] * np.exp(beta[1]*x)


res = odr(f, beta0, y, x, lower=lower, upper=upper, fjacb=fjacb, fjacd=fjacd,
          job=20, iprint=1001)

print("\n success:", res.success)
