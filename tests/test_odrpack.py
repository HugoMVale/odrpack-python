import numpy as np
from odrpack import odr
from odrpack.__odrpack import diwinf, dwinf, workspace_dimensions


def test_workspace_dimensions():
    n = 10
    nq = 2
    m = 3
    npar = 5
    isodr = True
    dims = workspace_dimensions(n, m, npar, nq, isodr)
    assert dims == (770, 46)
    assert dims[1] == 20 + 2*npar + nq*(npar + m)


def test_diwinf():
    res = diwinf(m=10, npar=5, nq=2)
    assert len(res) == 23
    assert all(idx >= 0 for idx in res.values())


def test_dwinf():
    res = dwinf(n=10, m=2, npar=5, nq=2, ldwe=1, ld2we=1, isodr=True)
    assert len(res) == 52
    assert all(idx >= 0 for idx in res.values())


def test_dimensions():
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


def test_example5():
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
