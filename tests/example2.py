import numpy as np

from odrpack import odr

beta0 = np.array([-1.0, -3.0, 0.09, 0.02, 0.08])
x = [[0.50, -0.12],
     [1.20, -0.60],
     [1.60, -1.00],
     [1.86, -1.40],
     [2.12, -2.54],
     [2.36, -3.36],
     [2.44, -4.00],
     [2.36, -4.75],
     [2.06, -5.25],
     [1.74, -5.64],
     [1.34, -5.97],
     [0.90, -6.32],
     [-0.28, -6.44],
     [-0.78, -6.44],
     [-1.36, -6.41],
     [-1.90, -6.25],
     [-2.50, -5.88],
     [-2.88, -5.50],
     [-3.18, -5.24],
     [-3.44, -4.86]]
x = np.array(x).T
y = np.full(x.shape[-1], 4.0)


def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    v, h = x
    return beta[2]*(v-beta[0])**2 + 2*beta[3]*(v-beta[0])*(h-beta[1]) \
        + beta[4]*(h-beta[1])**2 - 1


sol = odr(f, beta0, y, x, iprint=1001)

print("\n beta:", sol.beta)
print("\n delta:", sol.delta)
