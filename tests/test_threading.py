"""Thread-safety tests for odrpack.

These tests are meant to detect problems that only show up when several threads
use the library at the same time, in particular on free-threaded CPython builds
(3.13t, 3.14t, ...). They are also meaningful on regular (GIL) builds: the
model callbacks are Python code (often sleeping), so the GIL is released while
a fit is in progress and other threads can enter the Fortran core. Hidden
global/static state in the Fortran code, or shared mutable state in the wrapper,
would therefore corrupt results there as well.

What is checked:

1. The extension module declares itself free-threading-safe, i.e. importing it
   does not silently re-enable the GIL (free-threaded builds only).
2. Concurrent fits (heterogeneous problems, sleeping and non-sleeping
   callbacks) give bit-identical results to the same fits run serially.
3. Input arrays shared between threads are neither modified nor a source of
   cross-talk.
4. Re-entrancy: a fit can be started from inside the model function of another
   fit (this fails if the Fortran core keeps any static work storage).
5. Exceptions raised in the callbacks stay confined to the thread that raised
   them.
6. Concurrent report/error files (Fortran I/O units) do not get mixed up.
7. The small helper functions of the extension module are thread-safe.
"""
import os
import subprocess
import sys
import sysconfig
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from odrpack import OdrStop, odr_fit
from odrpack.__odrpack import loc_iwork, loc_rwork, stop_message, workspace_dimensions

N_THREADS = 8
N_ROUNDS = 3
DELAY = 0.0002  # s; makes the model function release the GIL on regular builds

FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


# %% Problem definitions (module-level, deterministic)

def _make_cases():
    cases = {}

    # m=1, q=1, analytic Jacobians, with weights
    def f1(x, beta):
        time.sleep(DELAY)
        return beta[0] + beta[1] * x + beta[2] * x**2 + beta[3] * x**3

    def jb1(x, beta):
        return np.stack([np.ones_like(x), x, x**2, x**3])[:, None, :]

    def jx1(x, beta):
        return (beta[1] + 2 * beta[2] * x + 3 * beta[3] * x**2)[None, None, :]

    x = np.linspace(-10., 10., 21)
    beta_true = np.array([1., -2., 0.1, -0.1])
    rng = np.random.default_rng(1)
    y = f1(x, beta_true) + rng.normal(0, 0.05, x.shape)
    cases["m1q1_jac"] = dict(
        f=f1, xdata=x, ydata=y, beta0=np.ones(4), jac_beta=jb1, jac_x=jx1,
        weight_x=0.5, weight_y=2.0)

    # m=2, q=1, finite differences, central scheme, bounds
    def f2(x, beta):
        time.sleep(DELAY)
        return (beta[0] * x[0, :])**3 + x[1, :]**beta[1]

    x = np.linspace(-10., 10., 41)
    x = np.vstack((x, 10 + x / 2))
    y = f2(x, np.array([2., 2.]))
    cases["m2q1_bounds"] = dict(
        f=f2, xdata=x, ydata=y, beta0=np.array([1.5, 1.5]),
        bounds=(np.array([0.5, 0.5]), np.array([3., 3.])),
        diff_scheme="central")

    # m=3, q=2, OLS-free ODR with fixed parameter
    def f3(x, beta):
        time.sleep(DELAY)
        y = np.zeros((2, x.shape[-1]))
        y[0, :] = (beta[0] * x[0, :])**3 + x[1, :]**beta[1] + np.exp(x[2, :] / 2)
        y[1, :] = (beta[2] * x[0, :])**2 + x[1, :]**beta[1]
        return y

    x = np.linspace(-1., 1., 31)
    x = np.vstack((x, np.exp(x), x**2))
    y = f3(x, np.array([1., 2., 3.]))
    cases["m3q2_fix"] = dict(
        f=f3, xdata=x, ydata=y, beta0=np.ones(3),
        fix_beta=np.array([False, True, False]))

    # OLS, no sleep (pure compute, maximises true parallelism)
    def f4(x, beta):
        return beta[0] * np.exp(beta[1] * x)

    x = np.linspace(0., 2., 200)
    rng = np.random.default_rng(2)
    y = f4(x, np.array([1.5, 0.7])) * (1 + 0.01 * rng.uniform(-1, 1, x.shape))
    cases["ols_nosleep"] = dict(
        f=f4, xdata=x, ydata=y, beta0=np.array([1., 1.]), task="OLS")

    # implicit ODR (ellipse), no sleep
    def f5(x, beta):
        return beta[0] * x[0]**2 + 2 * beta[1] * x[0] * x[1] + beta[2] * x[1]**2 - 1

    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    rng = np.random.default_rng(3)
    x = np.vstack((2 * np.cos(t) + rng.normal(0, 0.01, t.shape),
                   1 * np.sin(t) + rng.normal(0, 0.01, t.shape)))
    cases["implicit"] = dict(
        f=f5, xdata=x, ydata=np.zeros(t.size), beta0=np.array([0.3, 0.0, 0.8]),
        task="implicit-ODR")

    return cases


CASES = _make_cases()
RESULT_FIELDS = ("beta", "delta", "eps", "sd_beta", "cov_beta", "xplusd", "yest")
SCALAR_FIELDS = ("info", "nfev", "njev", "niter", "irank", "res_var",
                 "sum_square", "sum_square_delta", "sum_square_eps")


def _run(name, **overrides):
    kwargs = {**CASES[name], **overrides}
    return odr_fit(**kwargs)


def _assert_identical(sol, ref, label=""):
    for field in RESULT_FIELDS:
        a, b = getattr(sol, field), getattr(ref, field)
        assert np.array_equal(a, b, equal_nan=True), f"{label}: `{field}` differs"
    for field in SCALAR_FIELDS:
        a, b = getattr(sol, field), getattr(ref, field)
        assert a == b, f"{label}: `{field}` differs ({a} != {b})"


@pytest.fixture(scope="module")
def reference():
    """Serial reference solutions."""
    return {name: _run(name) for name in CASES}


def _run_in_threads(tasks, n_threads=N_THREADS):
    """Run callables in a pool of `n_threads` threads.

    The first `n_threads` tasks are released simultaneously by a barrier, to
    maximise the overlap between fits; the remaining tasks just queue up.
    """
    n_start = min(n_threads, len(tasks))
    barrier = threading.Barrier(n_start)
    lock = threading.Lock()
    counter = [0]

    def wrapper(task):
        with lock:
            first_batch = counter[0] < n_start
            counter[0] += 1
        if first_batch:
            barrier.wait(timeout=60)
        return task()

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(wrapper, t) for t in tasks]
        return [f.result() for f in futures]


# %% 1. Module declares free-threading support

@pytest.mark.skipif(not FREE_THREADED_BUILD, reason="needs a free-threaded build of CPython")
def test_import_does_not_enable_gil():
    """Importing odrpack must not silently re-enable the GIL.

    If the extension is not compiled as free-threading-compatible (nanobind needs
    `NB_FREE_THREADED`), CPython re-enables the GIL at import time and emits a
    RuntimeWarning. Everything would still 'work', but not free-threaded, and all
    the other tests in this module would be testing nothing.

    Runs in a fresh interpreter, in the default configuration (no `-X gil=0`,
    no PYTHON_GIL), which is what users get.
    """
    code = (
        "import sys\n"
        "assert not sys._is_gil_enabled()\n"
        "import odrpack, odrpack.__odrpack\n"
        "print('GIL_ENABLED=%s' % sys._is_gil_enabled())\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    proc = subprocess.run([sys.executable, "-W", "always", "-c", code],
                          capture_output=True, text=True, env=env)
    assert "GIL_ENABLED=False" in proc.stdout, (
        "odrpack re-enabled the GIL on import.\n" + proc.stdout + proc.stderr)
    assert "has not declared that it can run safely without the GIL" not in proc.stderr


@pytest.mark.skipif(not FREE_THREADED_BUILD, reason="needs a free-threaded build of CPython")
def test_gil_still_disabled_in_this_process():
    """The GIL must not have been re-enabled by importing/using odrpack here."""
    _run("ols_nosleep")
    assert not sys._is_gil_enabled()


# %% 2. Concurrent fits == serial fits

@pytest.mark.parametrize("name", list(CASES))
def test_same_problem_in_many_threads(name, reference):
    """Many threads solving the *same* problem at the same moment."""
    results = _run_in_threads([lambda: _run(name)] * (N_THREADS * 2))
    for i, sol in enumerate(results):
        _assert_identical(sol, reference[name], f"{name}[{i}]")


def test_different_problems_in_many_threads(reference):
    """Heterogeneous problems (different n, m, q, npar, options) at the same time.

    Catches static/global work storage in the Fortran core, which would be
    clobbered by problems of different size.
    """
    names = list(CASES) * N_ROUNDS
    tasks = [(lambda n=n: (n, _run(n))) for n in names]
    for name, sol in _run_in_threads(tasks):
        _assert_identical(sol, reference[name], name)


def test_threads_with_shared_inputs(reference):
    """All threads share the very same input arrays (and function objects)."""
    before = {k: {a: b.copy() for a, b in c.items() if isinstance(b, np.ndarray)}
              for k, c in CASES.items()}
    tasks = [(lambda n=n: (n, _run(n))) for n in list(CASES) * N_ROUNDS]
    for name, sol in _run_in_threads(tasks):
        _assert_identical(sol, reference[name], name)
    for k, arrays in before.items():
        for a, b in arrays.items():
            assert np.array_equal(CASES[k][a], b), f"input `{a}` of case `{k}` was modified"


def test_results_are_independent_objects():
    """Result arrays returned to different threads must not alias each other."""
    results = _run_in_threads([lambda: _run("ols_nosleep")] * N_THREADS)
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            for field in ("beta", "delta", "eps", "rwork", "iwork"):
                assert not np.shares_memory(getattr(a, field), getattr(b, field))
    # mutate one, the others must be unaffected
    ref = results[1].beta.copy()
    results[0].beta[:] = -1
    assert np.array_equal(results[1].beta, ref)


# %% 3. Re-entrancy

def test_nested_fit_inside_model_function(reference):
    """A fit started from inside the model function of another fit.

    Deterministic, single-threaded detector of static storage in the Fortran core.
    """
    inner_ref = reference["ols_nosleep"]
    base = CASES["m2q1_bounds"]
    inner_results = []

    def f_outer(x, beta):
        inner_results.append(_run("ols_nosleep"))
        return base["f"](x, beta)

    sol = odr_fit(**{**base, "f": f_outer})
    _assert_identical(sol, reference["m2q1_bounds"], "outer")
    assert len(inner_results) > 0
    for s in inner_results:
        _assert_identical(s, inner_ref, "inner")


def test_nested_fit_inside_model_function_threaded(reference):
    def task():
        base = CASES["m3q2_fix"]
        inner = []

        def f_outer(x, beta):
            inner.append(_run("m1q1_jac"))
            return base["f"](x, beta)

        outer = odr_fit(**{**base, "f": f_outer})
        return outer, inner

    for outer, inner in _run_in_threads([task] * N_THREADS):
        _assert_identical(outer, reference["m3q2_fix"], "outer")
        for s in inner:
            _assert_identical(s, reference["m1q1_jac"], "inner")


# %% 4. Exceptions stay in their thread

class _Boom(Exception):
    pass


def test_exceptions_do_not_leak_between_threads(reference):
    base = CASES["m2q1_bounds"]

    def make_failing(exc, after):
        calls = {"n": 0}

        def f(x, beta):
            calls["n"] += 1
            if calls["n"] > after:
                raise exc
            return base["f"](x, beta)
        return f

    def task_ok():
        return "ok", _run("m2q1_bounds")

    def task_boom():
        try:
            odr_fit(**{**base, "f": make_failing(_Boom("boom"), 5)})
        except _Boom:
            return "boom", None
        return "no-exception", None

    def task_stop():
        # `OdrStop` raised from within the iteration is handled by the library
        try:
            sol = odr_fit(**{**base, "f": make_failing(OdrStop("stop"), 5)})
        except OdrStop:
            return "odrstop-raised", None
        return "odrstop-handled", sol

    kinds = ["ok", "boom", "stop"]
    tasks = {"ok": task_ok, "boom": task_boom, "stop": task_stop}

    # serial expectation for the "stop" kind
    expected_stop = tasks["stop"]()[0]

    order = kinds * (N_THREADS * 2)
    results = _run_in_threads([tasks[k] for k in order])
    for k, (label, sol) in zip(order, results):
        if k == "ok":
            assert label == "ok"
            _assert_identical(sol, reference["m2q1_bounds"], "ok-thread")
        elif k == "boom":
            assert label == "boom"
        else:
            assert label == expected_stop


# %% 5. Fortran I/O units

def test_report_and_error_files_do_not_mix(tmp_path):
    """Every thread writes its own report/error file: contents must equal serial ones."""
    names = ["m1q1_jac", "m2q1_bounds", "m3q2_fix", "ols_nosleep"]

    def run_with_files(name, tag):
        rpt = tmp_path / f"{tag}_{name}.rpt"
        err = tmp_path / f"{tag}_{name}.err"
        sol = _run(name, report="long", rptfile=str(rpt), errfile=str(err))
        return rpt.read_text(), err.read_text(), sol

    ref = {n: run_with_files(n, "ref") for n in names}

    tasks = [(lambda n=n, i=i: (n, run_with_files(n, f"t{i}")))
             for i, n in enumerate(names * N_THREADS)]
    for name, (rpt, err, sol) in _run_in_threads(tasks):
        assert rpt == ref[name][0], f"report of `{name}` differs from serial one"
        assert err == ref[name][1], f"error file of `{name}` differs from serial one"
        assert len(rpt) > 0
        _assert_identical(sol, ref[name][2], name)


def test_shared_report_file_name_fails_cleanly(tmp_path):
    """Several threads use the same report file name (a user error).

    The Fortran runtime refuses to open one file in two units at once, which
    surfaces as a RuntimeError. The only requirements are: no crash/hang, no
    other exception type, and the library keeps working afterwards.
    """
    path = str(tmp_path / "shared.rpt")

    def task():
        try:
            return _run("ols_nosleep", report="short", rptfile=path)
        except RuntimeError:
            return None

    results = _run_in_threads([task] * N_THREADS)
    assert any(r is not None for r in results)
    assert all(r is None or r.success for r in results)
    # still healthy afterwards
    assert _run("ols_nosleep", report="short", rptfile=path).success


def test_report_to_stdout_from_many_threads(capfd, reference):
    """Concurrent use of the default Fortran output unit must not crash or corrupt results."""
    results = _run_in_threads([lambda: _run("ols_nosleep", report="short")] * N_THREADS)
    capfd.readouterr()
    for sol in results:
        _assert_identical(sol, reference["ols_nosleep"], "stdout-report")


# %% 6. Helper functions

def test_helper_functions_in_threads():
    ref_dims = workspace_dimensions(50, 2, 3, 4, True)
    ref_iw = loc_iwork(2, 3, 4)
    ref_rw = loc_rwork(50, 2, 3, 4, 1, 1, True)
    ref_msg = [stop_message(i) for i in (1, 2, 3, 4, 1000, 10000, 20000, 51000)]

    def task():
        for _ in range(200):
            assert workspace_dimensions(50, 2, 3, 4, True) == ref_dims
            assert loc_iwork(2, 3, 4) == ref_iw
            assert loc_rwork(50, 2, 3, 4, 1, 1, True) == ref_rw
            assert [stop_message(i) for i in (1, 2, 3, 4, 1000, 10000, 20000, 51000)] == ref_msg
        return True

    assert all(_run_in_threads([task] * N_THREADS))


# %% 7. Long-ish stress (many fits, more threads than cores)

def test_stress(reference):
    names = list(CASES)
    tasks = [(lambda n=names[i % len(names)]: (n, _run(n))) for i in range(64)]
    for name, sol in _run_in_threads(tasks, n_threads=16):
        _assert_identical(sol, reference[name], name)
