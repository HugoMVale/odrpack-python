import warnings
from typing import Callable

import numpy as np

from odrpack.__odrpack import loc_iwork, loc_rwork
from odrpack.__odrpack import odr as _odr
from odrpack.__odrpack import workspace_dimensions
from odrpack.result import F64Array, I32Array, OdrResult

__all__ = ['odr']


def odr(f: Callable[[F64Array, F64Array], F64Array],
        beta0: F64Array,
        y: F64Array,
        x: F64Array,
        *,
        we: float | F64Array | None = None,
        wd: float | F64Array | None = None,
        fjacb: Callable[[F64Array, F64Array], F64Array] | None = None,
        fjacd: Callable[[F64Array, F64Array], F64Array] | None = None,
        ifixb: I32Array | None = None,
        ifixx: I32Array | None = None,
        delta0: F64Array | None = None,
        lower: F64Array | None = None,
        upper: F64Array | None = None,
        job: int = 0,
        iprint: int | None = 0,
        rptfile: str | None = None,
        errfile: str | None = None,
        ndigit: int | None = None,
        taufac: float | None = None,
        sstol: float | None = None,
        partol: float | None = None,
        maxit: int | None = None,
        stpb: F64Array | None = None,
        stpd: F64Array | None = None,
        sclb: F64Array | None = None,
        scld: F64Array | None = None,
        rwork: F64Array | None = None,
        iwork: I32Array | None = None,
        ) -> OdrResult:
    r"""Solve a weighted orthogonal distance regression (ODR) problem, also
    known as errors-in-variables regression.

    Parameters
    ----------
    f : Callable[[F64Array, F64Array], F64Array]
        Function to be fitted, with the signature `f(beta, x)`. It must return
        an array with the same shape as `y`.
    beta0 : F64Array
        Array of shape `(npar,)` with the initial guesses of the model parameters,
        within the bounds specified by arguments `lower` and `upper` (if they are
        specified).
    y : F64Array
        Array of shape `(n,)` or `(q, n)` containing the values of the response
        variable(s). When the model is explicit, the user must specify a value
        for each element of `y`. If some responses of some observations are
        actually missing, then the user can set the corresponding weight in
        argument `we` to zero in order to remove the effect of the missing
        observation from the analysis. When the model is implicit, `y` is not
        referenced.
    x : F64Array
        Array of shape `(n,)` or `(m, n)` containing the values of the explanatory
        variable(s).
    we : float | F64Array | None
        Scalar or array specifying how the errors on `y` are to be weighted.
        If `we` is a scalar, then it is used for all data points. If `we` is
        an array of shape `(n,)` and `q==1`, then `we[i]` represents the weight
        for `y[i]`. If `we` is an array of shape `(q,)`, then it represents the
        diagonal of the covariant weighting matrix for all data points. If `we`
        is an array of shape `(q, q)`, then it represents the full covariant
        weighting matrix for all data points. If `we` is an array of shape
        `(q, n)`, then `we[:, i]` represents the diagonal of the covariant
        weighting matrix for `y[:, i]`. If `we` is an array of shape `(q, q, n)`,
        then `we[:, :, i]` represents the full covariant weighting matrix for
        `y[:, i]`. For a comprehensive description of the options, refer to page
        25 of the ODRPACK95 guide. By default, `we` is set to one for all `y`
        data points.
    wd : float | F64Array | None
        Scalar or array specifying how the errors on `x` are to be weighted.
        If `wd` is a scalar, then it is used for all data points. If `wd` is
        an array of shape `(n,)` and `m==1`, then `wd[i]` represents the weight
        for `x[i]`. If `wd` is an array of shape `(m,)`, then it represents the
        diagonal of the covariant weighting matrix for all data points. If `wd`
        is an array of shape `(m, m)`, then it represents the full covariant
        weighting matrix for all data points. If `wd` is an array of shape
        `(m, n)`, then `wd[:, i]` represents the diagonal of the covariant
        weighting matrix for `x[:, i]`. If `wd` is an array of shape `(m, m, n)`,
        then `wd[:, :, i]` represents the full covariant weighting matrix for
        `x[:, i]`. For a comprehensive description of the options, refer to page
        26 of the ODRPACK95 guide. By default, `wd` is set to one for all `x`
        data points.
    fjacb : Callable[[F64Array, F64Array], F64Array] | None
        Jacobian of the function to be fitted with respect to `beta`, with the
        signature `fjacb(beta, x)`. It must return an array with shape 
        `(q, npar, n)` or compatible. To activate this option, `job` must be
        set accordingly. By default, the Jacobian is evaluated numerically
        according to the finite difference scheme defined in `job`.
    fjacd : Callable[[F64Array, F64Array], F64Array] | None
        Jacobian of the function to be fitted with respect to `delta`, with the
        signature `fjacd(beta, x)`. It must return an array with shape 
        `(q, m, n)` or compatible. To activate this option, `job` must be
        set accordingly. By default, the Jacobian is evaluated numerically
        according to the finite difference scheme defined in `job`.
    ifixb : I32Array | None
        Array with the same shape as `beta0`, containing the values designating
        which elements of `beta` are to be held fixed. Zero means the parameter
        is held fixed, and one means it is adjustable. By default, `ifixb` is
        set to one for all elements of `beta`.
    ifixx : I32Array | None
        Array with the same shape as `x`, containing the values designating
        which elements of `x` are to be held fixed. Alternatively, it can be a
        rank-1 array of shape `(m,)` or `(n,)`, in which case it will be broadcast
        along the other axis. Zero means the element is held fixed and one means
        it is free. By default, in orthogonal distance regression mode, `ifixx`
        is set to one for all elements of `x`. In ordinary least squares mode,
        the `x` values are intrinsically fixed.
    delta0 : F64Array | None
        Array with the same shape as `x`, containing the initial guesses of the
        errors in the explanatory variable. To activate this option, `job` must
        be set accordingly. By default, `delta0` is set to zero for all elements
        of `x`.
    lower : F64Array | None
        Array with the same shape as `beta0`, containing the lower bounds of the
        model parameters. By default, `lower` is set to negative infinity for
        all elements of `beta`.
    upper : F64Array | None
        Array with the same shape as `beta0`, containing the upper bounds of the
        model parameters. By default, `upper` is set to positive infinity for
        all elements of `beta`.
    job : int
        Variable controlling problem initialization and computational method.
        The default value is 0, corresponding to an explicit orthogonal distance
        regression, with `delta0` initialized to zero, derivatives computed by
        forward finite difference, and covariance matrix computed using Jacobian
        matrices recomputed at the final solution. Another common option is 20,
        corresponding to an explicit orthogonal distance regression with 
        user-supplied jacobians `fjacb` and `fjacd`. For an implicit orthogonal
        distance regression, `job` must be set to 1. To initialize `delta0` with
        the user supplied values, the 4th digit of `job` must be set to 1, e.g.
        1000. To restart a previous run, the 5th digit of `job` must be set to
        1, e.g. 10000. For a comprehensive description of the options, refer to
        page 28 of the ODRPACK95 guide.
    iprint : int | None
        Variable controlling the generation of computation reports. By default,
        no reports are generated. Some common values are: 1001 - short initial
        and final summary; 2002 - long initial and final summary; 11j1 - short
        initial and final summary, and short iteration summary every `j`
        iterations. For a comprehensive description of the options, refer to
        page 30 of the ODRPACK95 guide.
    rptfile : str | None
        File name for storing the computation reports, as defined by `iprint`.
        By default, the reports are sent to standard output.
    errfile : str | None
        File name for storing the error reports, as defined by `iprint`. By
        default, the reports are sent to standard error.
    ndigit : int | None
        Number of reliable decimal digits in the values computed by the model
        function `f` and its Jacobians `fjacb`, and `fjacd`. By default, the
        value is numerically determined by evaluating `f`. 
    taufac : float | None
        Factor comprised between 0 and 1 to initialize the trust region radius.
        The default value is 1. Reducing `taufac` may be appropriate if, at the
        first iteration, the computed results for the full Gauss-Newton step
        cause an overflow, or cause `beta` and/or `delta` to leave the region
        of interest. 
    sstol : float | None
        Factor comprised between 0 and 1 specifying the stopping tolerance for
        the sum of the squares convergence. The default value is `eps**(1/2)`,
        where `eps` is the machine precision in `float64`.
    partol : float | None
        Factor comprised between 0 and 1 specifying the stopping tolerance for
        parameter convergence (i.e., `beta` and `delta`). When the model is
        explicit, the default value is `eps**(2/3)`, and when the model is
        implicit, the default value is `eps**(1/3)`, where `eps` is the machine
        precision in `float64`.
    maxit : int | None
        Maximum number of allowed iterations. The default value is 50 for a
        first run and 10 for a restart (see `job`).
    stpb : F64Array | None
        Array with the same shape as `beta0` containing the _relative_ step
        sizes used to compute the finite difference derivatives with respect
        to the model parameters. By default, `stpb` is set internally based on
        the value of `ndigit` and the type of finite differences used. For
        additional details, refer to pages 31 and 78 of the ODRPACK95 guide.
    stpd : F64Array | None
        Array with the same shape as `x`, containing the _relative_ step sizes
        used to compute the finite difference derivatives with respect to the
        errors in the explanatory variable. Alternatively, it can be a rank-1
        array of shape `(m,)` or `(n,)`, in which case it will be broadcast along
        the other axis. By default, `stpd` is set internally based on the value
        of `ndigit` and the type of finite differences used. For additional
        details, refer to pages 31 and 78 of the ODRPACK95 guide.
    sclb : F64Array | None
        Array with the same shape as `beta0` containing the scale values of the
        model parameters. Scaling is used to improve the numerical stability
        of the regression, but does not affect the problem specification. Scaling
        should not be confused with the weighting matrices `we` and `wd`. By
        default, `sclb` is set internally based on the relative magnitudes of 
        `beta`. For further details, refer to pages 32 and 84 of the ODRPACK95
        guide.
    scld : F64Array | None
        Array with the same shape as `x`, containing the scale values of the
        errors in the explanatory variable. Alternatively, it can be a rank-1
        array of shape `(m,)` or `(n,)`, in which case it will be broadcast along
        the other axis. Scaling is used to improve the numerical stability of
        the regression, but does not affect the problem specification. Scaling
        should not be confused with the weighting matrices `we` and `wd`. By
        default, `scld` is set internally based on the relative magnitudes of
        `x`. For further details, refer to pages 32 and 85 of the ODRPACK95 guide.
    rwork : F64Array | None
        Array containing the real-valued internal state of the odrpack solver.
        It is only required for a restart (see `job`), in which case it must be
        set to the state of the previous run.
    iwork : I32Array | None
        Array containing the integer-valued internal state of the odrpack solver.
        It is only required for a restart (see `job`), in which case it must be
        set to the state of the previous run.

    Returns
    -------
    OdrResult
        An object containing the results of the regression.


    References
    ----------

    [1] Jason W. Zwolak, Paul T. Boggs, and Layne T. Watson.
        Algorithm 869: ODRPACK95: A weighted orthogonal distance regression code 
        with bound constraints. ACM Trans. Math. Softw. 33, 4 (August 2007), 27-es.
        https://doi.org/10.1145/1268776.1268782

    [2] Jason W. Zwolak, Paul T. Boggs, and Layne T. Watson. User's Reference
        Guide for ODRPACK95, 2005.
        https://github.com/HugoMVale/odrpack95/blob/main/original/Doc/guide.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from odrpack import odr
    >>> x = np.array([0.982, 1.998, 4.978, 6.01])
    >>> y = np.array([2.7, 7.4, 148.0, 403.0])
    >>> beta0 = np.array([2., 0.5])
    >>> lower = np.array([0., 0.])
    >>> upper = np.array([10., 0.9])
    >>> def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    ...     return beta[0] * np.exp(beta[1]*x)
    >>> sol = odr(f, beta0, y, x, lower=lower, upper=upper, iprint=0)
    >>> sol.beta
    array([1.63336897, 0.9       ])
    """

    # Future deprecation warning
    warnings.warn(
        "This function is deprecated and will be removed in a future version. "
        "Please use `odr_fit` instead.",
        FutureWarning
    )

    # Interpret job
    is_odr = _get_digit(job, 1) < 2
    has_jac = _get_digit(job, 2) > 1
    has_delta0 = _get_digit(job, 4) > 0
    is_restart = _get_digit(job, 5) > 0

    # Check x and y
    if x.ndim == 1:
        m = 1
    elif x.ndim == 2:
        m = x.shape[0]
    else:
        raise ValueError(
            f"`x` must be a rank-1 array of shape `(n,)` or a rank-2 array of shape `(m, n)`, but has shape {x.shape}.")

    if y.ndim == 1:
        q = 1
    elif y.ndim == 2:
        q = y.shape[0]
    else:
        raise ValueError(
            f"`y` must be a rank-1 array of shape `(n,)` or a rank-2 array of shape `(q, n)`, but has shape {y.shape}.")

    if x.shape[-1] == y.shape[-1]:
        n = x.shape[-1]
    else:
        raise ValueError(
            f"The last dimension of `x` and `y` must be identical, but x.shape={x.shape} and y.shape={y.shape}.")

    # Check beta0 and related parameters
    if beta0.ndim == 1:
        npar = beta0.size
        beta = beta0.copy()
    else:
        raise ValueError(
            f"`beta0` must be a rank-1 array of shape `(npar,)`, but has shape {beta0.shape}.")

    if lower is not None:
        if lower.shape != beta0.shape:
            raise ValueError("`lower` must have the same shape as `beta0`.")
        if np.any(lower >= beta0):
            raise ValueError("`lower` must be less than `beta0`.")

    if upper is not None:
        if upper.shape != beta0.shape:
            raise ValueError("`upper` must have the same shape as `beta0`.")
        if np.any(upper <= beta0):
            raise ValueError("`upper` must be greater than `beta0`.")

    if ifixb is not None and ifixb.shape != beta0.shape:
        raise ValueError("`ifixb` must have the same shape as `beta0`.")

    if stpb is not None and stpb.shape != beta0.shape:
        raise ValueError("`stpb` must have the same shape as `beta0`.")

    if sclb is not None and sclb.shape != beta0.shape:
        raise ValueError("`sclb` must have the same shape as `beta0`.")

    # Check delta0
    if has_delta0 and delta0 is not None:
        if delta0.shape != x.shape:
            raise ValueError("`delta0` must have the same shape as `x`.")
        delta = delta0.copy()
    elif not has_delta0 and delta0 is None:
        delta = np.zeros(x.shape, dtype=np.float64)
    else:
        raise ValueError("Inconsistent arguments for `job` and `delta0`.")

    # Check ifixx
    if ifixx is not None:
        if ifixx.shape == x.shape:
            ldifx = n
        elif ifixx.shape == (m,) and m > 1 and n != m:
            ldifx = 1
        elif ifixx.shape == (n,) and m > 1 and n != m:
            ldifx = n
            ifixx = np.tile(ifixx, (m, 1))
        else:
            raise ValueError(
                "`ifixx` must either have the same shape as `x` or be a rank-1 array of shape `(m,)` or `(n,)`. See page 26 of the ODRPACK95 User Guide.")
    else:
        ldifx = 1

    # Check stpd
    if stpd is not None:
        if stpd.shape == x.shape:
            ldstpd = n
        elif stpd.shape == (m,) and m > 1 and n != m:
            ldstpd = 1
        elif stpd.shape == (n,) and m > 1 and n != m:
            ldstpd = n
            stpd = np.tile(stpd, (m, 1))
        else:
            raise ValueError(
                "`stpd` must either have the same shape as `x` or be a rank-1 array of shape `(m,)` or `(n,)`. See page 31 of the ODRPACK95 User Guide.")
    else:
        ldstpd = 1

    # Check scld
    if scld is not None:
        if scld.shape == x.shape:
            ldscld = n
        elif scld.shape == (m,) and m > 1 and n != m:
            ldscld = 1
        elif scld.shape == (n,) and m > 1 and n != m:
            ldscld = n
            scld = np.tile(scld, (m, 1))
        else:
            raise ValueError(
                "`scld` must either have the same shape as `x` or be a rank-1 array of shape `(m,)` or `(n,)`. See page 32 of the ODRPACK95 User Guide.")
    else:
        ldscld = 1

    # Check we
    if we is not None:
        if isinstance(we, (float, int)):
            ldwe = 1
            ld2we = 1
            we = np.full((q,), we, dtype=np.float64)
        elif isinstance(we, np.ndarray):
            if we.shape == (q,):
                ldwe = 1
                ld2we = 1
            elif we.shape == (q, q):
                ldwe = 1
                ld2we = q
            elif we.shape == (q, n) or (we.shape == (n,) and q == 1):
                ldwe = n
                ld2we = 1
            elif we.shape in ((q, 1, 1), (q, 1, n), (q, q, 1), (q, q, n)):
                ldwe = we.shape[2]
                ld2we = we.shape[1]
            else:
                raise ValueError(
                    r"`we` must be a array of shape `(q,)`, `(n,)`, `(q, q)`, `(q, n)`, `(q, 1, 1)`, `(q, 1, n)`, `(q, q, 1)`, or `(q, q, n)`. See page 25 of the ODRPACK95 User Guide.")
        else:
            raise TypeError("`we` must be a float or an array.")
    else:
        ldwe = 1
        ld2we = 1

    # Check wd
    if wd is not None:
        if isinstance(wd, (float, int)):
            ldwd = 1
            ld2wd = 1
            wd = np.full((m,), wd, dtype=np.float64)
        elif isinstance(wd, np.ndarray):
            if wd.shape == (m,):
                ldwd = 1
                ld2wd = 1
            elif wd.shape == (m, m):
                ldwd = 1
                ld2wd = m
            elif wd.shape == (m, n) or (wd.shape == (n,) and m == 1):
                ldwd = n
                ld2wd = 1
            elif wd.shape in ((m, 1, 1), (m, 1, n), (m, m, 1), (m, m, n)):
                ldwd = wd.shape[2]
                ld2wd = wd.shape[1]
            else:
                raise ValueError(
                    r"`wd` must be a array of shape `(m,)`, `(n,)`, `(m, m)`, `(m, n)`, `(m, 1, 1)`, `(m, 1, n)`, `(m, m, 1)`, or `(m, m, n)`. See page 26 of the ODRPACK95 User Guide.")
        else:
            raise TypeError("`wd` must be a float or an array.")
    else:
        ldwd = 1
        ld2wd = 1

    # Check model function and jacobians
    f0 = f(beta0, x)
    def f_(x, beta): return f(beta, x)

    if f0.shape != y.shape:
        raise ValueError(
            "Function `f` must return an array with the same shape as `y`.")

    def fdummy(x, beta): return np.array([np.nan])  # will never be called

    if has_jac and fjacb is not None:
        fjacb0 = fjacb(beta0, x)
        def fjacb_(x, beta): return fjacb(beta, x)
        if fjacb0.shape[-1] != n or fjacb0.size != n*npar*q:
            raise ValueError(
                "Function `fjacb` must return an array with shape `(n, npar, q)` or compatible.")
    elif not has_jac and fjacb is None:
        fjacb_ = fdummy
    else:
        raise ValueError("Inconsistent arguments for `job` and `fjacb`.")

    if has_jac and fjacd is not None:
        fjacd0 = fjacd(beta0, x)
        def fjacd_(x, beta): return fjacd(beta, x)
        if fjacd0.shape[-1] != n or fjacd0.size != n*m*q:
            raise ValueError(
                "Function `fjacd` must return an array with shape `(n, m, q)` or compatible.")
    elif not has_jac and fjacd is None:
        fjacd_ = fdummy
    else:
        raise ValueError("Inconsistent arguments for `job` and `fjacd`.")

    # Check/allocate work arrays
    lwork, liwork = workspace_dimensions(n, m, q, npar, is_odr)
    if (not is_restart) and (rwork is None) and (iwork is None):
        rwork = np.zeros(lwork, dtype=np.float64)
        iwork = np.zeros(liwork, dtype=np.int32)
    elif is_restart and (rwork is not None) and (iwork is not None):
        if rwork.size != lwork:
            raise ValueError(
                "Work array `work` does not have the correct length.")
        if iwork.size != liwork:
            raise ValueError(
                "Work array `iwork` does not have the correct length.")
    else:
        raise ValueError(
            "Inconsistent arguments for `job`, `work` and `iwork`.")

    # Call the ODRPACK95 routine
    # Note: beta, delta, work, and iwork are modified in place
    info = _odr(n=n, m=m, q=q, npar=npar,
                ldwe=ldwe, ld2we=ld2we,
                ldwd=ldwd, ld2wd=ld2wd,
                ldifx=ldifx,
                ldstpd=ldstpd, ldscld=ldscld,
                f=f_, fjacb=fjacb_, fjacd=fjacd_,
                beta=beta, y=y, x=x,
                delta=delta,
                we=we, wd=wd, ifixb=ifixb, ifixx=ifixx,
                lower=lower, upper=upper,
                rwork=rwork, iwork=iwork,
                job=job,
                ndigit=ndigit, taufac=taufac, sstol=sstol, partol=partol, maxit=maxit,
                iprint=iprint, errfile=errfile, rptfile=rptfile
                )

    # Indexes of integer and real work arrays
    iwork_idx: dict[str, int] = loc_iwork(m, q, npar)
    rwork_idx: dict[str, int] = loc_rwork(n, m, q, npar, ldwe, ld2we, is_odr)

    # Return the result
    # Extract results without messing up the original work arrays
    i0_eps = rwork_idx['eps']
    eps = rwork[i0_eps:i0_eps+y.size].copy()
    eps = np.reshape(eps, y.shape)

    i0_sd = rwork_idx['sd']
    sd_beta = rwork[i0_sd:i0_sd+beta.size].copy()

    i0_vcv = rwork_idx['vcv']
    cov_beta = rwork[i0_vcv:i0_vcv+beta.size**2].copy()
    cov_beta = np.reshape(cov_beta, (beta.size, beta.size))

    result = OdrResult(
        beta=beta,
        delta=delta,
        eps=eps,
        xplusd=x+delta,
        yest=y+eps,
        sd_beta=sd_beta,
        cov_beta=cov_beta,
        res_var=rwork[rwork_idx['rvar']],
        info=info,
        success=info < 4,
        nfev=iwork[iwork_idx['nfev']],
        njev=iwork[iwork_idx['njev']],
        niter=iwork[iwork_idx['niter']],
        irank=iwork[iwork_idx['irank']],
        inv_condnum=rwork[rwork_idx['rcond']],
        sum_square=rwork[rwork_idx['wss']],
        sum_square_delta=rwork[rwork_idx['wssdel']],
        sum_square_eps=rwork[rwork_idx['wsseps']],
        iwork=iwork,
        rwork=rwork,
    )

    return result


def _get_digit(number: int, ndigit: int) -> int:
    """Return the n-th digit from the right of `number`."""
    return (number // 10**(ndigit-1)) % 10
