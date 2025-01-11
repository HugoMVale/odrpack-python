import os

import numpy as np
import pytest

from odrpack import odr
from odrpack.__odrpack import diwinf, dwinf, workspace_dimensions

SEED = 1234567890


def test_diwinf():
    res = diwinf(m=10, npar=5, nq=2)
    assert len(res) == 23
    assert all(idx >= 0 for idx in res.values())


def test_dwinf():
    res = dwinf(n=10, m=2, npar=5, nq=2, ldwe=1, ld2we=1, isodr=True)
    assert len(res) == 52
    assert all(idx >= 0 for idx in res.values())


def test_workspace_dimensions():
    n = 10
    nq = 2
    m = 3
    npar = 5
    isodr = True
    dims = workspace_dimensions(n, m, npar, nq, isodr)
    assert dims == (770, 46)
    assert dims[1] == 20 + 2*npar + nq*(npar + m)


def test_dimension_consistency():
    n = 11
    nq = 2
    m = 3
    npar = 5
    for isodr in [True, False]:
        dims = workspace_dimensions(n, m, npar, nq, isodr)
        iworkidx = diwinf(m, npar, nq)
        workidx = dwinf(n, m, npar, nq, ldwe=1, ld2we=1, isodr=isodr)
        assert dims[0] >= workidx['lwkmn']
        assert dims[1] >= iworkidx['liwkmn']


@pytest.fixture
def case1():
    # m=1, nq=1
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
    # m=2, nq=1
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
    # m=3, nq=2
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


def test_parameters(case1):
    # maxit
    sol = odr(**case1, maxit=2)
    assert sol.info == 4
    assert 'iteration limit' in sol.stopreason.lower()

    # sstol
    sstol = 0.123
    sol = odr(**case1, sstol=sstol)
    assert sol.info == 1
    idxwork = dwinf(case1['x'].size, 1, case1['beta0'].size, 1, 1, 1, True)
    assert np.isclose(sol.work[idxwork['sstol']], sstol)

    # partol
    partol = 0.456
    sol = odr(**case1, partol=partol)
    assert sol.info == 2
    assert np.isclose(sol.work[idxwork['partl']], partol)

    # taufac
    taufac = 0.6969
    sol = odr(**case1, taufac=taufac)
    assert sol.info == 1
    assert np.isclose(sol.work[idxwork['taufc']], taufac)


def test_restart(case1):

    # valid restart
    sol_ref = odr(**case1)
    sol1 = odr(**case1, maxit=2)
    sol2 = odr(**case1, job=10_000, iwork=sol1.iwork, work=sol1.work)
    assert sol2.info == 1
    assert np.allclose(sol_ref.beta, sol2.beta)

    # invalid restarts
    with pytest.raises(ValueError):
        _ = odr(**case1, iwork=sol1.iwork, work=sol1.work)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, iwork=sol1.iwork)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, work=sol1.work)
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, iwork=sol1.iwork, work=np.ones(10000))
    with pytest.raises(ValueError):
        _ = odr(**case1, job=10_000, iwork=np.ones(10000, dtype=np.int32), work=sol1.work)


def test_rptfile_and_errfile(case1):

    # write to report file
    rptfile = 'rtptest.txt'
    for iprint, rptsize in zip([0, 1001], [0, 2600]):
        if os.path.isfile(rptfile):
            os.remove(rptfile)
        sol = odr(**case1, iprint=iprint, rptfile=rptfile)
        assert os.path.isfile(rptfile) \
            and abs(os.path.getsize(rptfile) - rptsize) < 200

    # write to error file
    errfile = 'errtest.txt'
    if os.path.isfile(errfile):
        os.remove(errfile)
    sol = odr(**case1, iprint=1001,
              errfile=errfile)
    assert os.path.isfile(errfile)  # and os.path.getsize(errfile) > 0

    # write to report and error file
    if os.path.isfile(rptfile):
        os.remove(rptfile)
    if os.path.isfile(errfile):
        os.remove(errfile)
    sol = odr(**case1, job=10, iprint=1001,
              rptfile=rptfile,
              errfile=errfile)
    assert os.path.isfile(rptfile) and os.path.getsize(rptfile) > 2500
    assert os.path.isfile(errfile)  # and os.path.getsize(errfile) > 0

    # I can't get the error file to be written to..


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


def add_noise(array, noise, seed):
    """Adds random noise to an array."""
    rng = np.random.default_rng(seed)
    return array*(1 + noise*rng.uniform(-1, 1, size=array.shape))
