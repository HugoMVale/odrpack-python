from odrpack import odr_fit
import numpy as np

xdata = [0.982, 1.998, 4.978, 6.01]
ydata = [2.7, 7.4, 148.0, 403.0]

beta0 = [2.0, 0.5]
bounds = ([0.0, 0.0], [10.0, 0.9])


def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    "Model function."
    return beta[0] * np.exp(beta[1]*x)


sol = odr_fit(f, xdata, ydata, beta0, bounds=bounds)

print("beta:", sol.beta)
print("delta:", sol.delta)
