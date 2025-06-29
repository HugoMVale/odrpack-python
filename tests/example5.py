import numpy as np

from odrpack import odr_fit

beta0 = np.array([2., 0.5])
lower = np.array([0., 0.])
upper = np.array([10., 0.9])
xdata = np.array([0.982, 1.998, 4.978, 6.01])
ydata = np.array([2.7, 7.4, 148.0, 403.0])


def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] * np.exp(beta[1]*x)


def jac_beta(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    jac = np.zeros((beta.size, x.size))
    jac[0, :] = np.exp(beta[1]*x)
    jac[1, :] = beta[0]*x*np.exp(beta[1]*x)
    return jac


def jac_x(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] * beta[1] * np.exp(beta[1]*x)


sol = odr_fit(f, xdata, ydata, beta0, bounds=(lower, upper),
              jac_beta=jac_beta, jac_x=jac_x, report='short')

print("\n beta:", sol.beta)
print("\n delta:", sol.delta)
