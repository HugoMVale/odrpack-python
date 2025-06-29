import os
from copy import deepcopy

import numpy as np
import pytest

from odrpack import odr
from odrpack.__odrpack import loc_rwork

SEED = 1234567890


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

    return {'x': x, 'y': y, 'f': f, 'beta0': np.zeros_like(beta_star)}


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

    return {'x': x, 'y': y, 'f': f, 'beta0': np.array([1., 1.])}


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

    return {'x': x, 'y': y, 'f': f, 'beta0': np.array([5., 5., 5.])}


def test_base_cases(case1, case2, case3):

    # case 1
    sol1 = odr(**case1)
    assert sol1.success
    assert sol1.info == 1

    # case 2
    sol2 = odr(**case2)
    assert sol2.success
    assert sol2.info == 1

    # case 3
    sol3 = odr(**case3)
    assert sol3.success
    assert sol3.info == 1

    # invalid inputs:
    with pytest.raises(ValueError):
        # x and y don't have the same size
        _ = odr(f=case1['f'], x=np.ones(case1['x'].size+1), y=case1['y'],
                beta0=case1['beta0'])
    with pytest.raises(ValueError):
        # x has invalid shape
        _ = odr(f=case2['f'], x=np.ones((1, 2, 10)), y=np.ones(10),
                beta0=case2['beta0'])
    with pytest.raises(ValueError):
        # y has invalid shape
        _ = odr(f=case2['f'], x=np.ones((2, 10)), y=np.ones((1, 2, 10)),
                beta0=case2['beta0'])


def test_beta0_related(case1):

    # reference
    sol1 = odr(**case1)

    # fix some parameters
    sol = odr(**case1, ifixb=np.array([0, 1, 1, 0], dtype=np.int32))
    assert np.isclose(sol.beta[0], 0) and np.isclose(sol.beta[-1], 0)

    # fix all parameters
    sol = odr(**case1, ifixb=np.array([0, 0, 0, 0], dtype=np.int32))
    assert np.allclose(sol.beta, [0]*4)

    # user-defined stpb
    sol = odr(**case1, stpb=np.full(4, 1e-5))
    assert sol.info == 1
    assert np.allclose(sol.beta, sol1.beta)

    # user-defined sclb
    sol = odr(**case1, sclb=np.array([2, 2, 20, 20]))
    assert sol.info == 1
    assert np.allclose(sol.beta, sol1.beta)

    # invalid inputs:
    with pytest.raises(ValueError):
        # lower > beta0
        lower = case1['beta0'].copy()
        lower[1:] -= 1
        _ = odr(**case1, lower=lower)
    with pytest.raises(ValueError):
        # upper < beta0
        upper = case1['beta0'].copy()
        upper[1:] += 1
        _ = odr(**case1, upper=upper)
    with pytest.raises(ValueError):
        # beta0 has invalid shape
        _ = odr(f=case1['f'], x=case1['x'], y=case1['y'],
                beta0=np.zeros((4, 1)))
    with pytest.raises(ValueError):
        # beta0 has invalid shape
        _ = odr(f=case1['f'], x=case1['x'], y=case1['y'],
                beta0=np.zeros((1, 4)))
    with pytest.raises(ValueError):
        # lower has invalid shape
        _ = odr(**case1, lower=np.zeros((1, 4)))
    with pytest.raises(ValueError):
        # upper has invalid shape
        _ = odr(**case1, upper=np.zeros((1, 4)))
    with pytest.raises(ValueError):
        # ifixb has invalid shape
        _ = odr(**case1, ifixb=np.array([0, 1, 0], dtype=np.int32))
    with pytest.raises(ValueError):
        # stpb has invalid shape
        _ = odr(**case1, stpb=np.array([1e-4, 1., 2.]))
    with pytest.raises(ValueError):
        # sclb has invalid shape
        _ = odr(**case1, sclb=np.array([1., 1., 1., 1., 1.]))


def test_delta0_related(case1, case3):

    # user-defined delta0
    sol = odr(**case1, job=1010, delta0=np.ones_like(case1['x']))
    assert sol.info == 1

    # fix some x
    ifixx = np.ones_like(case1['x'], dtype=np.int32)
    fix = (4, 8)
    ifixx[fix,] = 0
    sol = odr(**case1, ifixx=ifixx)
    assert np.allclose(sol.delta[fix,], [0, 0])

    # fix some x, broadcast (m,)
    ifixx = np.ones(case3['x'].shape[0], dtype=np.int32)
    fix = (1)
    ifixx[fix,] = 0
    sol = odr(**case3, ifixx=ifixx)
    assert np.allclose(sol.delta[fix, :], np.zeros(sol.delta.shape[1]))

    # fix some x, broadcast (n,)
    ifixx = np.ones(case3['x'].shape[-1], dtype=np.int32)
    fix = (2, 7, 13)
    ifixx[fix,] = 0
    sol = odr(**case3, ifixx=ifixx)
    assert np.allclose(sol.delta[:, fix], np.zeros((sol.delta.shape[0], len(fix))))

    # fix all x (n,)
    ifixx = np.zeros_like(case1['x'], dtype=np.int32)
    sol = odr(**case1, ifixx=ifixx)
    assert np.allclose(sol.delta, np.zeros_like(sol.delta))

    # fix all x (m, n)
    ifixx = np.zeros_like(case3['x'], dtype=np.int32)
    sol = odr(**case3, ifixx=ifixx)
    assert np.allclose(sol.delta, np.zeros_like(sol.delta))

    # user stpd
    sol3 = odr(**case3)
    for shape in [case3['x'].shape, case3['x'].shape[0], case3['x'].shape[-1]]:
        stpd = np.full(shape, 1e-5)
        sol = odr(**case3, stpd=stpd)
        assert np.allclose(sol.delta, sol3.delta)

    # user scld
    sol3 = odr(**case3)
    for shape in [case3['x'].shape, case3['x'].shape[0], case3['x'].shape[-1]]:
        scld = np.full(shape, 10.)
        sol = odr(**case3, scld=scld)
        assert np.allclose(sol.delta, sol3.delta)

    # invalid inputs
    with pytest.raises(ValueError):
        # ifixx has invalid shape
        _ = odr(**case1, ifixx=np.array([0, 1, 0], dtype=np.int32))
    with pytest.raises(ValueError):
        # stpd has invalid shape
        _ = odr(**case3, stpd=np.array([1e-4, 1.]))
    with pytest.raises(ValueError):
        # scld has invalid shape
        _ = odr(**case3, scld=np.array([1., 1., 1., 1.]))
    with pytest.raises(ValueError):
        # delta0 has invalid shape
        _ = odr(**case3, job=1000, delta0=np.zeros_like(case1['y']))
    with pytest.raises(ValueError):
        # delta0 with wrong job
        _ = odr(**case1, job=100, delta0=np.zeros_like(case1['x']))


def test_wd(case1, case3):

    # wd scalar
    sol = odr(**case1, wd=1e10)
    assert np.allclose(sol.delta, np.zeros_like(sol.delta))

    # wd (n,) and m==1
    wd = np.ones_like(case1['x'])
    fix = (4, 7)
    wd[fix,] = 1e10
    sol = odr(**case1, wd=wd)
    assert np.allclose(sol.delta[fix,], np.zeros_like(sol.delta[fix,]))

    # wd (m, n)
    wd = np.ones_like(case3['x'])
    fix = (4, 13)
    wd[:, fix,] = 1e10
    sol = odr(**case3, wd=wd)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.delta[:, fix,], np.zeros((sol.delta.shape[0], len(fix))))

    # wd (m, 1, n)
    wd = np.expand_dims(wd, axis=1)
    sol = odr(**case3, wd=wd)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # wd (m,)
    wd = np.ones(case3['x'].shape[0])
    fix = (1,)
    wd[fix,] = 1e10
    sol = odr(**case3, wd=wd)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.delta[fix, :], np.zeros((len(fix), sol.delta.shape[-1])))

    # wd (m, 1, 1)
    wd = wd[..., np.newaxis, np.newaxis]
    sol = odr(**case3, wd=wd)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # wd (m, m)
    m = case3['x'].shape[0]
    wd = np.zeros((m, m))
    np.fill_diagonal(wd, 1.)
    fix = (1,)
    wd[fix, fix] = 1e10
    sol = odr(**case3, wd=wd)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.delta[fix, :], np.zeros((len(fix), sol.delta.shape[-1])))

    # wd (m, m, 1)
    wd = wd[..., np.newaxis]
    sol = odr(**case3, wd=wd)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # wd (m, m, n)
    wd = np.tile(wd, (1, 1, case3['x'].shape[-1]))
    sol = odr(**case3, wd=wd)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # wd has invalid shape
    wd = np.ones((1, 1, 1))
    with pytest.raises(ValueError):
        _ = odr(**case3, wd=wd)

    # wd has invalid tye
    with pytest.raises(TypeError):
        _ = odr(**case3, wd=[1., 1., 1.])


def test_we(case1, case3):

    ATOL = 1e-6

    # we scalar
    sol = odr(**case1, we=1e10)
    assert np.allclose(sol.eps, np.zeros_like(sol.eps))

    # we (n,) and q==1
    we = np.ones_like(case1['y'])
    fix = (4, 7)
    we[fix,] = 1e10
    sol = odr(**case1, we=we)
    assert np.allclose(sol.eps[fix,], np.zeros_like(sol.eps[fix,]), atol=ATOL)

    # we (q, n)
    we = np.ones_like(case3['y'])
    fix = (4, 13)
    we[:, fix,] = 1e10
    sol = odr(**case3, we=we)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.eps[:, fix,], np.zeros((sol.eps.shape[0], len(fix))),
                       atol=ATOL)

    # we (q, 1, n)
    we = np.expand_dims(we, axis=1)
    sol = odr(**case3, we=we)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # we (q,)
    we = np.ones(case3['y'].shape[0])
    fix = (1,)
    we[fix,] = 1e10
    sol = odr(**case3, we=we)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.eps[fix, :], np.zeros((len(fix), sol.eps.shape[-1])),
                       atol=ATOL)

    # we (q, 1, 1)
    we = we[..., np.newaxis, np.newaxis]
    sol = odr(**case3, we=we)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # we (q, q)
    q = case3['y'].shape[0]
    we = np.zeros((q, q))
    np.fill_diagonal(we, 1.)
    fix = (1,)
    we[fix, fix] = 1e10
    sol = odr(**case3, we=we)
    sol1 = deepcopy(sol)
    assert np.allclose(sol.eps[fix, :], np.zeros((len(fix), sol.eps.shape[-1])),
                       atol=ATOL)

    # we (q, q, 1)
    we = we[..., np.newaxis]
    sol = odr(**case3, we=we)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # we (q, q, n)
    we = np.tile(we, (1, 1, case3['y'].shape[-1]))
    sol = odr(**case3, we=we)
    assert np.allclose(sol.delta, sol1.delta)
    assert np.allclose(sol.eps, sol1.eps)

    # we has invalid shape
    we = np.ones((1, 1, 1))
    with pytest.raises(ValueError):
        _ = odr(**case3, we=we)

    # we has invalid tye
    with pytest.raises(TypeError):
        _ = odr(**case3, we=[1., 1., 1.])


def test_parameters(case1):
    # maxit
    sol = odr(**case1, maxit=2)
    assert sol.info == 4
    assert 'iteration limit' in sol.stopreason.lower()

    # sstol
    sstol = 0.123
    sol = odr(**case1, sstol=sstol)
    assert sol.info == 1
    rwork_idx = loc_rwork(case1['x'].size, 1, 1, case1['beta0'].size, 1, 1, True)
    assert np.isclose(sol.rwork[rwork_idx['sstol']], sstol)

    # partol
    partol = 0.456
    sol = odr(**case1, partol=partol)
    assert sol.info == 2
    assert np.isclose(sol.rwork[rwork_idx['partol']], partol)

    # taufac
    taufac = 0.6969
    sol = odr(**case1, taufac=taufac)
    assert sol.info == 1
    assert np.isclose(sol.rwork[rwork_idx['taufac']], taufac)


def test_restart(case1):

    # valid restart
    sol_ref = odr(**case1)
    sol1 = odr(**case1, maxit=2)
    sol2 = odr(**case1, job=10_000, iwork=sol1.iwork, rwork=sol1.rwork)
    assert sol2.info == 1
    assert np.allclose(sol_ref.beta, sol2.beta)

    # invalid restarts
    with pytest.raises(ValueError):
        _ = odr(**case1, iwork=sol1.iwork, rwork=sol1.rwork)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, iwork=sol1.iwork)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, rwork=sol1.rwork)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, iwork=sol1.iwork, rwork=np.ones(10000))
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, iwork=np.ones(10000, dtype=np.int32), rwork=sol1.rwork)


def test_rptfile_and_errfile(case1):

    rptfile = 'rtptest.txt'
    errfile = 'errtest.txt'

    # write to report file
    for iprint, rptsize in zip([0, 1001], [0, 2600]):
        if os.path.isfile(rptfile):
            os.remove(rptfile)
        _ = odr(**case1, iprint=iprint, rptfile=rptfile)
        assert os.path.isfile(rptfile) \
            and abs(os.path.getsize(rptfile) - rptsize) < 200

    # write to error file
    if os.path.isfile(errfile):
        os.remove(errfile)
    _ = odr(**case1, iprint=1001, errfile=errfile)
    assert os.path.isfile(errfile)  # and os.path.getsize(errfile) > 0

    # write to report and error file
    if os.path.isfile(rptfile):
        os.remove(rptfile)
    if os.path.isfile(errfile):
        os.remove(errfile)
    _ = odr(**case1, job=10, iprint=1001, rptfile=rptfile, errfile=errfile)
    assert os.path.isfile(rptfile) and os.path.getsize(rptfile) > 2500
    assert os.path.isfile(errfile)  # and os.path.getsize(errfile) > 0

    # I can't get the error file to be written to..

    # Clean up
    if os.path.isfile(rptfile):
        os.remove(rptfile)
    if os.path.isfile(errfile):
        os.remove(errfile)


def test_jacobians():

    # model and data are from odrpack's example5
    x = np.array([0.982, 1.998, 4.978, 6.01])
    y = np.array([2.7, 7.4, 148.0, 403.0])
    beta0 = np.array([2., 0.5])
    lower = np.array([0., 0.])
    upper = np.array([10., 0.9])

    def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return beta[0] * np.exp(beta[1]*x)

    def fjacb(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        jac = np.zeros((beta.size, x.size))
        jac[0, :] = np.exp(beta[1]*x)
        jac[1, :] = beta[0]*x*np.exp(beta[1]*x)
        return jac

    def fjacd(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return beta[0] * beta[1] * np.exp(beta[1]*x)

    beta_ref = np.array([1.63337602, 0.9])
    delta_ref = np.array([-0.36886137, -0.31273038, 0.029287, 0.11031505])

    # without jacobian
    for job in [0, 10]:
        sol = odr(f, beta0, y, x, lower=lower, upper=upper, job=job)
        assert np.allclose(sol.beta, beta_ref, rtol=1e-4)
        assert np.allclose(sol.delta, delta_ref, rtol=1e-3)

    # with jacobian
    sol = odr(f, beta0, y, x, lower=lower, upper=upper,
              fjacb=fjacb, fjacd=fjacd, job=20)
    assert np.allclose(sol.beta, beta_ref, rtol=1e-4)
    assert np.allclose(sol.delta, delta_ref, rtol=1e-3)

    # invalid f shape
    with pytest.raises(ValueError):
        _ = odr(lambda beta, x: np.reshape(f(beta, x), (-1, 1)),
                beta0, y, x)
    # invalid fjacb shape
    with pytest.raises(ValueError):
        _ = odr(f, beta0, y, x, job=20,
                fjacb=lambda beta, x: np.reshape(fjacb(beta, x), (-1, 1)),
                fjacd=fjacd)
    # invalid fjacd shape
    with pytest.raises(ValueError):
        _ = odr(f, beta0, y, x, job=20,
                fjacb=fjacb,
                fjacd=lambda beta, x: np.reshape(fjacd(beta, x), (-1, 1)))
    # missing fjacb
    with pytest.raises(ValueError):
        _ = odr(f, beta0, y, x, job=20,
                fjacd=fjacd)
    # missing fjacd
    with pytest.raises(ValueError):
        _ = odr(f, beta0, y, x, job=20,
                fjacb=fjacb)
    # with correct jacobian, but wrong job
    with pytest.raises(ValueError):
        _ = odr(f, beta0, y, x, job=0, fjacb=fjacb, fjacd=fjacd)


def test_implicit_model():

    # model and data are from odrpack's example2
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
    y = np.full(x.shape[-1], 0.0)

    def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
        v, h = x
        return beta[2]*(v-beta[0])**2 + 2*beta[3]*(v-beta[0])*(h-beta[1]) \
            + beta[4]*(h-beta[1])**2 - 1

    beta_ref = np.array([-9.99380462E-01,
                         -2.93104890E+00,
                         8.75730642E-02,
                         1.62299601E-02,
                         7.97538109E-02])

    sol = odr(f, beta0, y, x, job=1, wd=1)
    assert np.allclose(sol.beta, beta_ref)


def add_noise(array, noise, seed):
    """Adds random noise to an array."""
    rng = np.random.default_rng(seed)
    return array*(1 + noise*rng.uniform(-1, 1, size=array.shape))
