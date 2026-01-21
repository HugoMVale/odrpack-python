import os
from copy import deepcopy

import numpy as np
import pytest
from scipy.odr import odr as odrscipy
from scipy.optimize import curve_fit

from odrpack import OdrStop, odr_fit
from odrpack.__odrpack import loc_rwork

RNG = np.random.default_rng(seed=1234567890)


def add_noise(array, noise):
    """Adds random noise to an array."""
    return array*(1 + noise*RNG.uniform(-1, 1, size=array.shape))


def flipargs(f):
    """Flips the order of the arguments of a function."""
    return lambda x, beta: f(beta, x)


@pytest.fixture
def case1():
    "Made up test case with m=1, q=1"
    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return beta[0] + beta[1] * x + beta[2] * x**2 + beta[3] * x**3

    beta_star = np.array([1.0, -2.0, 0.1, -0.1])
    x = np.linspace(-10.0, 10.0, 21)
    y = f(x, beta_star)

    x = add_noise(x, 5e-2)
    y = add_noise(y, 10e-2)

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': np.zeros_like(beta_star)}


@pytest.fixture
def case2():
    "Made up test with m=2, q=1"
    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return (beta[0] * x[0, :])**3 + x[1, :]**beta[1]

    beta_star = np.array([2.0, 2.0])
    x1 = np.linspace(-10.0, 10.0, 41)
    x = np.vstack((x1, 10+x1/2))
    y = f(x, beta_star)

    x = add_noise(x, 5e-2)
    y = add_noise(y, 10e-2)

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': np.ones_like(beta_star)}


@pytest.fixture
def case3():
    "Made up test case with m=3, q=2"
    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        y = np.zeros((2, x.shape[-1]))
        y[0, :] = (beta[0] * x[0, :])**3 + x[1, :]**beta[1] + np.exp(x[2, :]/2)
        y[1, :] = (beta[2] * x[0, :])**2 + x[1, :]**beta[1]
        return y

    beta_star = np.array([1.0, 2.0, 3.0])
    x1 = np.linspace(-1.0, 1.0, 31)
    x = np.vstack((x1, np.exp(x1), x1**2))
    y = f(x, beta_star)

    x = add_noise(x, 5e-2)
    y = add_noise(y, 10e-2)

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': np.full_like(beta_star, 5.0)}


@pytest.fixture
def example2():
    "odrpack's example2"
    x = np.array([[0.50, -0.12],
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
                  [-3.44, -4.86]]).T
    y = np.full(x.shape[-1], 0.0)
    beta0 = np.array([-1.0, -3.0, 0.09, 0.02, 0.08])

    beta_ref = np.array([-9.99380462E-01,
                         -2.93104890E+00,
                         8.75730642E-02,
                         1.62299601E-02,
                         7.97538109E-02])

    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        v, h = x
        return beta[2]*(v-beta[0])**2 + 2*beta[3]*(v-beta[0])*(h-beta[1]) \
            + beta[4]*(h-beta[1])**2 - 1

    return {'xdata': x, 'ydata': y, 'f': f, 'beta0': beta0,
            'beta_ref': beta_ref}


@pytest.fixture
def example5():
    "odrpack's example5"
    x = np.array([0.982, 1.998, 4.978, 6.01])
    y = np.array([2.7, 7.4, 148.0, 403.0])
    beta0 = np.array([2.0, 0.5])
    bounds = (np.array([0.0, 0.0]), np.array([10.0, 0.9]))

    beta_ref = np.array([1.63337602, 0.9])
    delta_ref = np.array([-0.36886137, -0.31273038, 0.029287, 0.11031505])

    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return beta[0] * np.exp(beta[1]*x)

    def jac_beta(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        jac = np.zeros((beta.size, x.size))
        jac[0, :] = np.exp(beta[1]*x)
        jac[1, :] = beta[0]*x*np.exp(beta[1]*x)
        return jac

    def jac_x(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return beta[0] * beta[1] * np.exp(beta[1]*x)

    return {'f': f, 'xdata': x, 'ydata': y, 'beta0': beta0, 'bounds': bounds,
            'jac_beta': jac_beta, 'jac_x': jac_x, 'beta_ref': beta_ref,
            'delta_ref': delta_ref}


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
    with pytest.raises(ValueError):
        # x has nan values
        xdata = case1['xdata'].copy()
        xdata[0] = np.nan
        _ = odr_fit(f=case1['f'], xdata=xdata, ydata=case1['ydata'],
                    beta0=case1['beta0'])
    with pytest.raises(ValueError):
        # y has nan values
        ydata = case1['ydata'].copy()
        ydata[-1] = np.nan
        _ = odr_fit(f=case1['f'], xdata=case1['xdata'], ydata=ydata,
                    beta0=case1['beta0'])


def test_beta0_related(case1):

    # reference
    sol1 = odr_fit(**case1)

    # fix some parameters
    sol = odr_fit(**case1,
                  fix_beta=np.array([True, False, False, True]))
    assert np.isclose(sol.beta[0], 0) and np.isclose(sol.beta[-1], 0)

    # fix all parameters
    sol = odr_fit(**case1,
                  fix_beta=np.array([True, True, True, True]))
    assert np.allclose(sol.beta, np.zeros_like(sol.beta))

    # user-defined step_beta
    sol = odr_fit(**case1,
                  step_beta=np.full(4, 1e-5))
    assert sol.info == 1
    assert np.allclose(sol.beta, sol1.beta)

    # user-defined scale_beta
    sol = odr_fit(**case1,
                  scale_beta=np.array([2, 2, 20, 20]))
    assert sol.info == 1
    assert np.allclose(sol.beta, sol1.beta)

    # invalid inputs:
    with pytest.raises(ValueError):
        # lower > beta0
        lower = case1['beta0'].copy()
        lower[1:] -= 1
        _ = odr_fit(**case1, bounds=(lower, None))
    with pytest.raises(ValueError):
        # upper < beta0
        upper = case1['beta0'].copy()
        upper[1:] += 1
        _ = odr_fit(**case1, bounds=(None, upper))
    with pytest.raises(ValueError):
        # beta0 has invalid shape
        _ = odr_fit(f=case1['f'], xdata=case1['xdata'], ydata=case1['ydata'],
                    beta0=np.zeros((4, 1)))
    with pytest.raises(ValueError):
        # beta0 has invalid shape
        _ = odr_fit(f=case1['f'], xdata=case1['xdata'], ydata=case1['ydata'],
                    beta0=np.zeros((1, 4)))
    with pytest.raises(ValueError):
        # lower has invalid shape
        _ = odr_fit(**case1, bounds=(np.zeros((1, 4)), None))
    with pytest.raises(ValueError):
        # upper has invalid shape
        _ = odr_fit(**case1, bounds=(None, np.zeros((1, 4))))
    with pytest.raises(ValueError):
        # fix_beta has invalid shape
        _ = odr_fit(**case1, fix_beta=np.array([True, False, True]))
    with pytest.raises(ValueError):
        # step_beta has invalid shape
        _ = odr_fit(**case1, step_beta=np.array([1e-4, 1.0, 2.0]))
    with pytest.raises(ValueError):
        # scale_beta has invalid shape
        _ = odr_fit(**case1, scale_beta=np.ones(5))
    with pytest.raises(ValueError):
        # invalid task
        _ = odr_fit(**case1, task='invalid')
    with pytest.raises(ValueError):
        # beta0 has nan values
        beta0 = case1['beta0'].copy()
        beta0[0] = np.nan
        _ = odr_fit(f=case1['f'], xdata=case1['xdata'], ydata=case1['ydata'],
                    beta0=beta0)
    with pytest.raises(ValueError):
        # lower has nan values
        lower = case1['beta0'].copy() - 1.0
        lower[-1] = np.nan
        _ = odr_fit(**case1, bounds=(lower, None))
    with pytest.raises(ValueError):
        # upper has nan values
        upper = case1['beta0'].copy() + 1.0
        upper[0] = np.nan
        _ = odr_fit(**case1, bounds=(None, upper))
    with pytest.raises(ValueError):
        # fix_beta has nan values
        fix_beta = np.array([True, np.nan, False, False])
        _ = odr_fit(**case1, fix_beta=fix_beta)
    with pytest.raises(ValueError):
        # step_beta has nan values
        step_beta = case1['beta0'].copy()
        step_beta[2] = np.nan
        _ = odr_fit(**case1, step_beta=step_beta)
    with pytest.raises(ValueError):
        # scale_beta has nan values
        scale_beta = case1['beta0'].copy()
        scale_beta[2] = np.nan
        _ = odr_fit(**case1, scale_beta=scale_beta)


def test_delta0_related(case1, case3):

    # user-defined delta0
    sol = odr_fit(**case1, delta0=np.ones_like(case1['xdata']))
    assert sol.info == 1

    # fix some x
    fix_x = np.zeros_like(case1['xdata'], dtype=np.bool_)
    fix = (4, 8)
    fix_x[fix,] = True
    sol = odr_fit(**case1, fix_x=fix_x)
    assert np.allclose(sol.delta[fix,], [0, 0])

    # fix some x, broadcast (m,)
    fix_x = np.zeros(case3['xdata'].shape[0], dtype=np.bool_)
    fix = (1)
    fix_x[fix,] = True
    sol = odr_fit(**case3, fix_x=fix_x)
    assert np.allclose(sol.delta[fix, :], np.zeros_like(sol.delta[fix, :]))

    # fix some x, broadcast (n,)
    fix_x = np.zeros(case3['xdata'].shape[-1], dtype=np.bool_)
    fix = (2, 7, 13)
    fix_x[fix,] = True
    sol = odr_fit(**case3, fix_x=fix_x)
    assert np.allclose(sol.delta[:, fix], np.zeros_like(sol.delta[:, fix]))

    # fix all x (n,)
    fix_x = np.ones_like(case1['xdata'], dtype=np.bool_)
    sol = odr_fit(**case1, fix_x=fix_x)
    assert np.allclose(sol.delta, np.zeros_like(sol.delta))

    # fix all x (m, n)
    fix_x = np.ones_like(case3['xdata'], dtype=np.bool_)
    sol = odr_fit(**case3, fix_x=fix_x)
    assert np.allclose(sol.delta, np.zeros_like(sol.delta))

    # user step_delta
    sol3 = odr_fit(**case3)
    for shape in [case3['xdata'].shape,
                  case3['xdata'].shape[0],
                  case3['xdata'].shape[-1]]:
        step_delta = np.full(shape, 1e-5)
        sol = odr_fit(**case3, step_delta=step_delta)
        assert np.allclose(sol.delta, sol3.delta, atol=1e-4)

    # user scale_delta
    sol3 = odr_fit(**case3)
    for shape in [case3['xdata'].shape,
                  case3['xdata'].shape[0],
                  case3['xdata'].shape[-1]]:
        scale_delta = np.full(shape, 10.)
        sol = odr_fit(**case3, scale_delta=scale_delta)
        assert np.allclose(sol.delta, sol3.delta, atol=1e-4)

    # invalid inputs
    with pytest.raises(ValueError):
        # fix_x has invalid shape
        _ = odr_fit(**case1, fix_x=np.array([True, False, True]))
    with pytest.raises(ValueError):
        # step_delta has invalid shape
        _ = odr_fit(**case3, step_delta=np.array([1e-4, 1.0]))
    with pytest.raises(ValueError):
        # scale_delta has invalid shape
        _ = odr_fit(**case3, scale_delta=np.ones(4))
    with pytest.raises(ValueError):
        # delta0 has invalid shape
        _ = odr_fit(**case3, delta0=np.zeros_like(case1['ydata']))
    with pytest.raises(ValueError):
        # fix_x has nan values
        fix_x = np.ones_like(case1['xdata'])
        fix_x[0] = np.nan
        _ = odr_fit(**case1, fix_x=fix_x)
    with pytest.raises(ValueError):
        # step_delta has nan values
        step_delta = np.ones_like(case3['xdata'])
        step_delta[0, 0] = np.nan
        _ = odr_fit(**case3, step_delta=step_delta)
    with pytest.raises(ValueError):
        # scale_delta has nan values
        scale_delta = np.ones_like(case3['xdata'])
        scale_delta[1, 1] = np.nan
        _ = odr_fit(**case3, scale_delta=scale_delta)
    with pytest.raises(ValueError):
        # delta0 has nan values
        delta0 = np.ones_like(case3['xdata'])
        delta0[2, 2] = np.nan
        _ = odr_fit(**case3, delta0=delta0)


def test_weight_x(case1, case3):

    # weight_x scalar
    sol = odr_fit(**case1, weight_x=1e100)
    assert np.allclose(sol.delta, np.zeros_like(sol.delta))

    # weight_x (n,) and m=1
    weight_x = np.ones_like(case1['xdata'])
    fix = (4, 7)
    weight_x[fix,] = 1e100
    sol = odr_fit(**case1, weight_x=weight_x)
    assert np.allclose(sol.delta[fix,], np.zeros_like(sol.delta[fix,]))

    # weight_x (m, n)
    weight_x = np.ones_like(case3['xdata'])
    fix = (4, 13)
    weight_x[:, fix,] = 1e100
    sol = odr_fit(**case3, weight_x=weight_x)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.delta[:, fix,], np.zeros_like(sol.delta[:, fix,]))

    # weight_x (m, 1, n)
    weight_x = np.expand_dims(weight_x, axis=1)
    sol = odr_fit(**case3, weight_x=weight_x)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_x (m,)
    weight_x = np.ones(case3['xdata'].shape[0])
    fix = (1,)
    weight_x[fix,] = 1e100
    sol = odr_fit(**case3, weight_x=weight_x)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.delta[fix, :], np.zeros_like(sol.delta[fix, :]))

    # weight_x (m, 1, 1)
    weight_x = weight_x[..., np.newaxis, np.newaxis]
    sol = odr_fit(**case3, weight_x=weight_x)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_x (m, m)
    m = case3['xdata'].shape[0]
    weight_x = np.zeros((m, m))
    np.fill_diagonal(weight_x, 1.0)
    fix = (1,)
    weight_x[fix, fix] = 1e100
    sol = odr_fit(**case3, weight_x=weight_x)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.delta[fix, :], np.zeros_like(sol.delta[fix, :]))

    # weight_x (m, m, 1)
    weight_x = weight_x[..., np.newaxis]
    sol = odr_fit(**case3, weight_x=weight_x)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_x (m, m, n)
    weight_x = np.tile(weight_x, (1, 1, case3['xdata'].shape[-1]))
    sol = odr_fit(**case3, weight_x=weight_x)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_x has invalid shape
    weight_x = np.ones((1, 1, 1))
    with pytest.raises(ValueError):
        _ = odr_fit(**case3, weight_x=weight_x)

    # weight_x has invalid tye
    with pytest.raises(TypeError):
        _ = odr_fit(**case3, weight_x=[1.0, 1.0, 1.0])

    # weight_x has nan values
    weight_x = np.ones_like(case1['xdata'])
    weight_x[0] = np.nan
    with pytest.raises(ValueError):
        _ = odr_fit(**case1, weight_x=weight_x)


def test_weight_y(case1, case3):

    ATOL = 1e-6

    # weight_y scalar
    sol = odr_fit(**case1, weight_y=1e100)
    assert np.allclose(sol.eps, np.zeros_like(sol.eps))

    # weight_y (n,) and q=1
    weight_y = np.ones_like(case1['ydata'])
    fix = (4, 7)
    weight_y[fix,] = 1e100
    sol = odr_fit(**case1, weight_y=weight_y)
    assert np.allclose(sol.eps[fix,], np.zeros_like(sol.eps[fix,]), atol=ATOL)

    # weight_y (q, n)
    weight_y = np.ones_like(case3['ydata'])
    fix = (4, 13)
    weight_y[:, fix,] = 1e100
    sol = odr_fit(**case3, weight_y=weight_y)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.eps[:, fix,], np.zeros_like(sol.eps[:, fix,]), atol=ATOL)

    # weight_y (q, 1, n)
    weight_y = np.expand_dims(weight_y, axis=1)
    sol = odr_fit(**case3, weight_y=weight_y)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_y (q,)
    weight_y = np.ones(case3['ydata'].shape[0])
    fix = (1,)
    weight_y[fix,] = 1e100
    sol = odr_fit(**case3, weight_y=weight_y)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.eps[fix, :], np.zeros_like(sol.eps[fix, :]), atol=ATOL)

    # weight_y (q, 1, 1)
    weight_y = weight_y[..., np.newaxis, np.newaxis]
    sol = odr_fit(**case3, weight_y=weight_y)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_y (q, q)
    q = case3['ydata'].shape[0]
    weight_y = np.zeros((q, q))
    np.fill_diagonal(weight_y, 1.0)
    fix = (1,)
    weight_y[fix, fix] = 1e100
    sol = odr_fit(**case3, weight_y=weight_y)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.eps[fix, :], np.zeros_like(sol.eps[fix, :]), atol=ATOL)

    # weight_y (q, q, 1)
    weight_y = weight_y[..., np.newaxis]
    sol = odr_fit(**case3, weight_y=weight_y)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_y (q, q, n)
    weight_y = np.tile(weight_y, (1, 1, case3['ydata'].shape[-1]))
    sol = odr_fit(**case3, weight_y=weight_y)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # weight_y has invalid shape
    weight_y = np.ones((1, 1, 1))
    with pytest.raises(ValueError):
        _ = odr_fit(**case3, weight_y=weight_y)

    # weight_y has invalid type
    with pytest.raises(TypeError):
        _ = odr_fit(**case3, weight_y=[1.0, 1.0, 1.0])

    # weight_y has nan values
    weight_y = np.ones_like(case1['ydata'])
    weight_y[-1] = np.nan
    with pytest.raises(ValueError):
        _ = odr_fit(**case1, weight_y=weight_y)


def test_parameters(case1):
    # maxit
    sol = odr_fit(**case1, maxit=2)
    assert sol.info == 4
    assert 'iteration limit' in sol.stopreason.lower()

    # sstol
    sstol = 0.123
    sol = odr_fit(**case1, sstol=sstol)
    assert sol.info == 1
    rwork_idx = loc_rwork(case1['xdata'].size, 1, 1,
                          case1['beta0'].size, 1, 1, True)
    assert np.isclose(sol.rwork[rwork_idx['sstol']], sstol)

    # partol
    partol = 0.456
    sol = odr_fit(**case1, partol=partol)
    assert sol.info == 2
    assert np.isclose(sol.rwork[rwork_idx['partol']], partol)

    # taufac
    taufac = 0.6969
    sol = odr_fit(**case1, taufac=taufac)
    assert sol.info == 1
    assert np.isclose(sol.rwork[rwork_idx['taufac']], taufac)


def test_rptfile_and_errfile(case1):

    rptfile = 'rtptest.txt'
    errfile = 'errtest.txt'

    # write to report file
    for report, rptsize in zip(['none', 'short'], [0, 2600]):
        if os.path.isfile(rptfile):
            os.remove(rptfile)
        _ = odr_fit(**case1, report=report, rptfile=rptfile)
        assert os.path.isfile(rptfile) \
            and abs(os.path.getsize(rptfile) - rptsize) < 200

    # write to error file
    if os.path.isfile(errfile):
        os.remove(errfile)
    _ = odr_fit(**case1, report='short', errfile=errfile)
    assert os.path.isfile(errfile)  # and os.path.getsize(errfile) > 0

    # write to report and error file
    if os.path.isfile(rptfile):
        os.remove(rptfile)
    if os.path.isfile(errfile):
        os.remove(errfile)
    _ = odr_fit(**case1, diff_scheme='central', report='short',
                rptfile=rptfile, errfile=errfile)
    assert os.path.isfile(rptfile) and os.path.getsize(rptfile) > 2500
    assert os.path.isfile(errfile)  # and os.path.getsize(errfile) > 0

    # I can't get the error file to be written to..

    # Clean up
    if os.path.isfile(rptfile):
        os.remove(rptfile)
    if os.path.isfile(errfile):
        os.remove(errfile)


def test_jacobians(example5):

    xdata = example5['xdata']
    ydata = example5['ydata']
    beta0 = example5['beta0']
    bounds = example5['bounds']
    f = example5['f']
    jac_beta = example5['jac_beta']
    jac_x = example5['jac_x']
    beta_ref = example5['beta_ref']
    delta_ref = example5['delta_ref']

    # ODR without jacobian
    for diff_scheme in ['forward', 'central']:
        sol = odr_fit(f, xdata, ydata, beta0, bounds=bounds,
                      diff_scheme=diff_scheme)
        assert np.allclose(sol.beta, beta_ref, rtol=1e-5)
        assert np.allclose(sol.delta, delta_ref, rtol=1e-4)

    # ODR with jacobian
    sol = odr_fit(f, xdata, ydata, beta0, bounds=bounds,
                  jac_beta=jac_beta, jac_x=jac_x)
    assert np.allclose(sol.beta, beta_ref, rtol=1e-6)
    assert np.allclose(sol.delta, delta_ref, rtol=1e-6)

    # OLS with jacobian
    sol1 = odr_fit(f, xdata, ydata, beta0, weight_x=1e100)
    sol2 = odr_fit(f, xdata, ydata, beta0, jac_beta=jac_beta, task='OLS')
    assert np.allclose(sol2.beta, sol1.beta)
    assert np.allclose(sol2.delta, np.zeros_like(xdata))

    # invalid f shape
    with pytest.raises(ValueError):
        _ = odr_fit(lambda beta, x: np.reshape(f(beta, x), (-1, 1)),
                    xdata, ydata, beta0)
    # invalid jac_beta shape
    with pytest.raises(ValueError):
        _ = odr_fit(f, xdata, ydata, beta0,
                    jac_beta=lambda beta, x: np.reshape(jac_beta(beta, x), (-1, 1)),
                    jac_x=jac_x)
    # invalid jac_x shape
    with pytest.raises(ValueError):
        _ = odr_fit(f, xdata, ydata, beta0,
                    jac_beta=jac_beta,
                    jac_x=lambda beta, x: np.reshape(jac_x(beta, x), (-1, 1)))
    # missing jac_beta
    with pytest.raises(ValueError):
        _ = odr_fit(f, xdata, ydata, beta0,
                    jac_x=jac_x)
    # missing jac_x
    with pytest.raises(ValueError):
        _ = odr_fit(f, xdata, ydata, beta0,
                    jac_beta=jac_beta)
    # invalid diff_scheme
    with pytest.raises(ValueError):
        _ = odr_fit(f, xdata, ydata, beta0, diff_scheme='invalid')


def test_implicit_model(example2):

    sol = odr_fit(example2['f'], example2['xdata'], example2['ydata'],
                  example2['beta0'], task='implicit-ODR')
    assert np.allclose(sol.beta, example2['beta_ref'])


def test_OLS(case1):

    sol1 = odr_fit(**case1, task='OLS')
    sol2 = odr_fit(**case1, weight_x=1e100)
    assert np.allclose(sol1.beta, sol2.beta)
    assert np.allclose(sol1.delta, np.zeros_like(sol1.delta))


def test_exception_odrstop():

    xdata = np.array([1.0, 2.0, 3.0, 4.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0])
    beta0 = np.array([1.0, 1.0])

    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        if beta[0] > 0:
            raise OdrStop("Oops!")
        return beta[0] * np.exp(beta[1]*x)

    with pytest.raises(OdrStop):
        sol = odr_fit(f, xdata, ydata, beta0)
        assert not sol.success
        assert sol.info == 51000


def test_compare_scipy(case1, case2, case3):

    # case1 // scipy.optimize.curve_fit
    sol1 = odr_fit(**case1, task='OLS')
    sol2 = curve_fit(lambda x, *b: case1['f'](x, np.array(b)),
                     case1['xdata'], case1['ydata'], case1['beta0'])
    assert np.allclose(sol1.beta, sol2[0], atol=1e-5)

    # case1,2,3 // scipy.odr.odr
    for case in [case1, case2, case3]:
        for subcase in range(2):
            if subcase == 0:
                wd = RNG.uniform(0.1, 1.0)
                we = RNG.uniform(0.1, 1.0)
            else:
                wd = RNG.uniform(0.1, 1.0, size=case['xdata'].shape)
                we = RNG.uniform(0.1, 1.0, size=case['ydata'].shape)

            sol1 = odr_fit(**case, weight_x=wd, weight_y=we)
            sol2 = odrscipy(flipargs(case['f']),
                            case['beta0'], case['ydata'], case['xdata'],
                            wd=wd, we=we, full_output=True)

            assert np.allclose(sol1.beta, sol2[0], rtol=1e-4)

            assert np.all(np.max(we*abs(sol1.eps - sol2[3]['eps']), -1) /
                          (np.max(case['ydata'], -1) - np.min(case['ydata'], -1)) < 1e-5)

            assert np.all(np.max(wd*abs(sol1.delta - sol2[3]['delta']), -1) /
                          (np.max(case['xdata'], -1) - np.min(case['xdata'], -1)) < 1e-4)
