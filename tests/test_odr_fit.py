from copy import deepcopy

import numpy as np
import pytest

from odrpack.odr_scipy import odr_fit

SEED = 1234567890


def add_noise(array, noise, seed):
    """Adds random noise to an array."""
    rng = np.random.default_rng(seed)
    return array*(1 + noise*rng.uniform(-1, 1, size=array.shape))


@pytest.fixture
def case1():
    # m=1, q=1
    def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return beta[0] + beta[1] * x + beta[2] * x**2 + beta[3] * x**3

    beta_star = np.array([1, -2., 0.1, -0.1])
    x = np.linspace(-10., 10., 21)
    y = f(beta_star, x)

    x = add_noise(x, 5e-2, SEED)
    y = add_noise(y, 10e-2, SEED)

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': np.zeros_like(beta_star)}


@pytest.fixture
def case2():
    # m=2, q=1
    def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return (beta[0] * x[0, :])**3 + x[1, :]**beta[1]

    beta_star = np.array([2., 2.])
    x1 = np.linspace(-10., 10., 41)
    x = np.vstack((x1, 10+x1/2))
    y = f(beta_star, x)

    x = add_noise(x, 5e-2, SEED)
    y = add_noise(y, 10e-2, SEED)

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': np.array([1., 1.])}


@pytest.fixture
def case3():
    # m=3, q=2
    def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        y = np.zeros((2, x.shape[-1]))
        y[0, :] = (beta[0] * x[0, :])**3 + x[1, :]**beta[1] + np.exp(x[2, :]/2)
        y[1, :] = (beta[2] * x[0, :])**2 + x[1, :]**beta[1]
        return y

    beta_star = np.array([1., 2., 3.])
    x1 = np.linspace(-1., 1., 31)
    x = np.vstack((x1, np.exp(x1), x1**2))
    y = f(beta_star, x)

    x = add_noise(x, 5e-2, SEED)
    y = add_noise(y, 10e-2, SEED)

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': np.array([5., 5., 5.])}


def test_base_cases(case1, case2, case3):

    # case 1
    sol1 = odr_fit(**case1)
    assert sol1.success
    assert sol1.info == 1

    # case 2
    sol2 = odr_fit(**case2)
    assert sol2.success
    assert sol2.info == 1

    # case 3
    sol3 = odr_fit(**case3)
    assert sol3.success
    assert sol3.info == 1

    # invalid inputs:
    with pytest.raises(ValueError):
        # x and y don't have the same size
        _ = odr_fit(f=case1['f'],
                    xdata=np.ones(case1['xdata'].size+1),
                    ydata=case1['ydata'],
                    beta0=case1['beta0'])
    with pytest.raises(ValueError):
        # x has invalid shape
        _ = odr_fit(f=case2['f'], xdata=np.ones((1, 2, 10)), ydata=np.ones(10),
                    beta0=case2['beta0'])
    with pytest.raises(ValueError):
        # y has invalid shape
        _ = odr_fit(f=case2['f'], xdata=np.ones((2, 10)), ydata=np.ones((1, 2, 10)),
                    beta0=case2['beta0'])
