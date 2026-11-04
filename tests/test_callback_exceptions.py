"""Exceptions raised in the user-supplied callbacks (model function and Jacobians).

The callbacks are invoked from Fortran. A C++ exception must never be unwound through
Fortran frames (not guaranteed to work, and fatal without unwind tables), so the extension
stores the exception, aborts the regression through the `istop` flag, and rethrows after
the Fortran routine has returned normally. These tests check the observable consequences.
"""
import traceback

import numpy as np
import pytest

from odrpack import OdrStop, odr_fit

XDATA = np.linspace(0, 2, 20)
YDATA = 1.5*np.exp(0.7*XDATA)
BETA0 = np.array([1., 1.])


def f_ok(x, beta):
    return beta[0]*np.exp(beta[1]*x)


def jb_ok(x, beta):
    return np.stack([np.exp(beta[1]*x), beta[0]*x*np.exp(beta[1]*x)])[:, None, :]


def jx_ok(x, beta):
    return (beta[0]*beta[1]*np.exp(beta[1]*x))[None, None, :]


class Counter:
    """Callable wrapper that counts calls and fails from the `after`-th one on.

    The first calls are done by `odr_fit` itself (argument checks), before entering
    Fortran, so `after` must be large enough for the failure to happen inside
    the iterations.
    """

    def __init__(self, fn, exc=None, after=6, value=None):
        self.fn, self.exc, self.after, self.value = fn, exc, after, value
        self.calls = 0

    def __call__(self, x, beta):
        self.calls += 1
        if self.calls > self.after:
            if self.exc is not None:
                raise self.exc
            return self.value
        return self.fn(x, beta)


class CustomError(Exception):
    pass


def _fit(f=f_ok, **kwargs):
    return odr_fit(f, XDATA, YDATA, BETA0, **kwargs)


def test_exception_type_message_and_traceback_are_preserved():
    def failing_model(x, beta):
        return model(x, beta)

    model = Counter(f_ok, CustomError("custom message"))
    with pytest.raises(CustomError, match="custom message") as excinfo:
        _fit(failing_model)
    tb = "".join(traceback.format_tb(excinfo.value.__traceback__))
    assert "failing_model" in tb  # the user's frame is in the traceback


@pytest.mark.parametrize("exc", [ValueError("v"), ZeroDivisionError("z"), KeyError("k"),
                                 KeyboardInterrupt(), CustomError("c")])
def test_various_exception_types_propagate(exc):
    with pytest.raises(type(exc)):
        _fit(Counter(f_ok, exc))


def test_no_python_callback_after_the_first_failure():
    """The optimiser must not keep calling the user's function once it has failed."""
    model = Counter(f_ok, CustomError("boom"))
    with pytest.raises(CustomError):
        _fit(model)
    assert model.calls == model.after + 1


def test_exception_in_jacobian_beta():
    jb = Counter(jb_ok, CustomError("jb"), after=3)
    with pytest.raises(CustomError, match="jb"):
        _fit(f_ok, jac_beta=jb, jac_x=jx_ok)
    assert jb.calls == jb.after + 1


def test_exception_in_jacobian_x():
    jx = Counter(jx_ok, CustomError("jx"), after=3)
    with pytest.raises(CustomError, match="jx"):
        _fit(f_ok, jac_beta=jb_ok, jac_x=jx)
    assert jx.calls == jx.after + 1


def test_wrong_return_type_in_callback_raises_instead_of_crashing():
    """A conversion error (a C++ exception that is not a python_error) in the callback."""
    with pytest.raises(Exception) as excinfo:
        _fit(Counter(f_ok, value="not an array"))
    assert not isinstance(excinfo.value, OdrStop)


def test_files_are_closed_after_an_exception(tmp_path):
    """The Fortran units opened for report/error files must be released.

    Reusing the same file name would fail if the previous unit were still open.
    """
    rpt, err = str(tmp_path / "x.rpt"), str(tmp_path / "x.err")
    with pytest.raises(CustomError):
        _fit(Counter(f_ok, CustomError("boom")), report="short", rptfile=rpt, errfile=err)
    sol = _fit(report="short", rptfile=rpt, errfile=err)
    assert sol.success
    (tmp_path / "x.rpt").unlink()  # not locked (matters on Windows)


def test_library_is_healthy_after_an_exception():
    ref = _fit()
    for exc in (CustomError("a"), ValueError("b")):
        with pytest.raises(type(exc)):
            _fit(Counter(f_ok, exc))
        sol = _fit()
        assert np.array_equal(sol.beta, ref.beta)
        assert sol.info == ref.info and sol.nfev == ref.nfev


def test_odrstop_is_still_graceful(capfd):
    model = Counter(f_ok, OdrStop("stop"))
    sol = _fit(model)
    assert not sol.success
    assert sol.info // 10000 == 5


def test_odrstop_then_other_exception_in_same_fit_is_not_swallowed():
    """`OdrStop` first, anything else later: a later, different exception must still propagate
    (if the optimiser calls the function again after a graceful stop)."""
    state = {"n": 0}

    def model(x, beta):
        state["n"] += 1
        if state["n"] == 7:
            raise OdrStop("stop")
        if state["n"] > 7:
            raise CustomError("later")
        return f_ok(x, beta)

    try:
        sol = _fit(model)
    except CustomError:
        return
    assert not sol.success
