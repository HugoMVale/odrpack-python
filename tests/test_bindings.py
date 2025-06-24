import numpy as np

from odrpack.__odrpack import loc_iwork, loc_rwork, odr, workspace_dimensions


def test_loc_iwork():
    res = loc_iwork(m=10, q=2, npar=5)
    assert len(res) == 23
    assert all(idx >= 0 for idx in res.values())


def test_loc_rwork():
    res = loc_rwork(n=10, m=2, q=2, npar=5, ldwe=1, ld2we=1, isodr=True)
    assert len(res) == 52
    assert all(idx >= 0 for idx in res.values())


def test_workspace_dimensions():
    n = 10
    q = 2
    m = 3
    npar = 5
    isodr = True
    lrwork, liwork = workspace_dimensions(n, m, q, npar, isodr)
    assert lrwork == 770
    assert liwork == 46


def test_dimension_consistency():
    n = 11
    q = 2
    m = 3
    npar = 5
    for isodr in [True, False]:
        dims = workspace_dimensions(n, m, q, npar, isodr)
        iwork_idx = loc_iwork(m, q, npar)
        rwork_idx = loc_rwork(n, m, q, npar, ldwe=1, ld2we=1, isodr=isodr)
        assert dims[0] >= rwork_idx['lrwkmn']
        assert dims[1] >= iwork_idx['liwkmn']


def test_odr():
    "example5 from odrpack"

    def f(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return beta[0] * np.exp(beta[1]*x)

    def fjacb(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        jac = np.zeros((beta.size, x.size))
        jac[0, :] = np.exp(beta[1]*x)
        jac[1, :] = beta[0]*x*np.exp(beta[1]*x)
        return jac

    def fjacd(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return beta[0] * beta[1] * np.exp(beta[1]*x)

    def fdummy(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
        return np.array([42.0])

    beta0 = np.array([2., 0.5])
    lower = np.array([0., 0.])
    upper = np.array([10., 0.9])
    x = np.array([0.982, 1.998, 4.978, 6.01])
    y = np.array([2.7, 7.4, 148.0, 403.0])
    n = x.size
    m = 1
    q = 1
    npar = beta0.size
    delta0 = np.zeros(x.shape)

    beta_ref = np.array([1.63337602E+00, 9.00000000E-01])

    # solution without jacobian
    beta = beta0.copy()
    delta = delta0.copy()
    info = odr(n, m, q, npar, 1, 1, 1, 1, 1, 1, 1,
               f, fdummy, fdummy, beta, y, x, delta,
               lower=lower, upper=upper, job=0)
    assert info == 1
    assert np.allclose(beta, beta_ref)

    # solution with jacobian
    beta = beta0.copy()
    delta = delta0.copy()
    info = odr(n, m, q, npar, 1, 1, 1, 1, 1, 1, 1,
               f, fjacb, fjacd, beta, y, x, delta,
               lower=lower, upper=upper, job=20)
    assert info == 1
    np.allclose(beta, beta_ref)
