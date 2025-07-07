import time
from multiprocessing.pool import Pool

import numpy as np

from odrpack import odr_fit

DELAY = 0.01  # s

# %% Functions need to be defined outside the test function


def f1(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x + beta[2] * x**2 + beta[3] * x**3


beta1 = np.array([1, -2., 0.1, -0.1])
x1 = np.linspace(-10., 10., 21, dtype=np.float64)
y1 = f1(x1, beta1)


def f2(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    time.sleep(np.random.uniform(0, DELAY))
    return (beta[0] * x[0, :])**3 + x[1, :]**beta[1]


beta2 = np.array([2., 2.])
x2 = np.linspace(-10., 10., 41, dtype=np.float64)
x2 = np.vstack((x2, 10+x2/2))
y2 = f2(x2, beta2)


def f3(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    time.sleep(np.random.uniform(0, DELAY))
    y = np.zeros((2, x.shape[-1]))
    y[0, :] = (beta[0] * x[0, :])**3 + x[1, :]**beta[1] + np.exp(x[2, :]/2)
    y[1, :] = (beta[2] * x[0, :])**2 + x[1, :]**beta[1]
    return y


beta3 = np.array([1., 2., 3.])
x3 = np.linspace(-1., 1., 31, dtype=np.float64)
x3 = np.vstack((x3, np.exp(x3), x3**2))
y3 = f3(x3, beta3)

case1 = (f1, x1, y1, np.ones_like(beta1))
case2 = (f2, x2, y2, np.ones_like(beta2))
case3 = (f3, x3, y3, np.ones_like(beta3))


def test_multiple_processes():

    # ref solutions
    sol1 = odr_fit(*case1)
    sol2 = odr_fit(*case2)
    sol3 = odr_fit(*case3)

    # multiple processes
    pool = Pool()
    num_jobs = 10
    cases = [case1, case2, case3]
    solutions = pool.starmap(odr_fit, cases*num_jobs)
    pool.close()
    pool.join()

    for i in range(0, len(solutions), len(cases)):
        assert np.allclose(solutions[i].beta, sol1.beta)
        assert np.allclose(solutions[i+1].beta, sol2.beta)
        assert np.allclose(solutions[i+2].beta, sol3.beta)
