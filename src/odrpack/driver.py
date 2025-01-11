from typing import Callable

import numpy as np
from numpy.typing import NDArray
from odrpack.__odrpack import diwinf, dwinf
from odrpack.__odrpack import odr as _odr
from odrpack.__odrpack import workspace_dimensions
from odrpack.result import OdrResult

__all__ = ['odr']

Float64Vector = NDArray[np.float64]
Float64Array = NDArray[np.float64]
Int32Vector = NDArray[np.int32]
Int32Array = NDArray[np.int32]


def odr(f: Callable[[Float64Vector, Float64Array], Float64Array],
        beta0: Float64Vector,
        y: Float64Array,
        x: Float64Array,
        *,
        we: float | Float64Array | None = None,
        wd: float | Float64Array | None = None,
        fjacb: Callable[[Float64Vector, Float64Array], Float64Array] | None = None,
        fjacd: Callable[[Float64Vector, Float64Array], Float64Array] | None = None,
        ifixb: Int32Vector | None = None,
        ifixx: Int32Array | None = None,
        delta0: Float64Array | None = None,
        lower: Float64Vector | None = None,
        upper: Float64Vector | None = None,
        job: int = 0,
        iprint: int | None = 0,
        rptfile: str | None = None,
        errfile: str | None = None,
        ndigit: int | None = None,
        taufac: float | None = None,
        sstol: float | None = None,
        partol: float | None = None,
        maxit: int | None = None,
        stpb: Float64Vector | None = None,
        stpd: Float64Array | None = None,
        sclb: Float64Vector | None = None,
        scld: Float64Array | None = None,
        work: Float64Vector | None = None,
        iwork: Int32Vector | None = None,
        ) -> OdrResult:
    r"""Solve a weighted orthogonal distance regression (ODR) problem, also
    known as errors-in-variables regression.

    Parameters
    ----------
    f : Callable[[Float64Vector, Float64Array], Float64Array]
        Function to be fitted, with the signature `f(beta, x)`. It must return
        an array with the same shape as `y`.
    beta0 : Float64Vector
        Rank-1 array of shape `(npar,)` with the initial guesses of the model
        parameters, within the bounds specified by `lower` and `upper` (if the
        bounds are specified).
    y : Float64Array
        Array of shape `(n,)` or `(nq, n)` containing the values of the response
        variable(s). When the model is explicit, the user must specify a value
        for each element of `y`. If some responses of some observations are
        actually missing, then the user can set the corresponding weight in
        argument `we` to zero in order to remove the effect of the missing
        observation from the analysis. When the model is implicit, `y` is not
        referenced.
    x : Float64Array
        Array of shape `(n,)` or `(m, n)` containing the values of the explanatory
        variable(s).
    we : float | Float64Array | None
        Scalar or array specifying how the errors on `y` are to be weighted.
        If `we` is an array, then it must be a rank-1 array of shape `(nq,)` or
        a rank-3 array of shape `(nq, ld2we, ldwe)`, where `ldwe ∈ {1, n}` and
        `ld2we ∈ {1, nq}`. For a comprehensive description of the options, refer
        to page 25 of the ODRPACK95 guide. By default, `we` is set to one for
        all `y` data points.
    wd : float | Float64Array | None
        Scalar or array specifying how the errors on `x` are to be weighted.
        If `wd` is an array, then it must be a rank-1 array of shape `(m,)` or
        a rank-3 array of shape `(m, ld2wd, ldwd)`, where `ldwd ∈ {1, n}` and
        `ld2wd ∈ {1, m}`. For a comprehensive description of the options, refer
        to page 26 of the ODRPACK95 guide. By default, `wd` is set to one for
        all `x` data points.
    fjacb : Callable[[Float64Vector, Float64Array], Float64Array] | None
        Jacobian of the function to be fitted with respect to `beta`, with the
        signature `fjacb(beta, x)`. It must return an array with shape 
        `(n, npar, nq)` or compatible. To activate this option, `job` must be
        set accordingly. By default, the Jacobian is evaluated numerically
        according to the finite difference scheme defined in `job`.
    fjacd : Callable[[Float64Vector, Float64Array], Float64Array] | None
        Jacobian of the function to be fitted with respect to `delta`, with the
        signature `fjacd(beta, x)`. It must return an array with shape 
        `(n, m, nq)` or compatible. To activate this option, `job` must be
        set accordingly. By default, the Jacobian is evaluated numerically
        according to the finite difference scheme defined in `job`.
    ifixb : Int32Vector | None
        Rank-1 array with the same shape as `beta0`, containing the values
        designating which elements of `beta` are to be held fixed. Zero means
        the parameter is held fixed, and one means it is adjustable. By default,
        `ifixb` is set to one for all elements of `beta`.
    ifixx : Int32Array | None
        Array with the same shape as `x`, containing the values designating
        which elements of `x` are to be held fixed. Alternatively, it can be a
        rank-1 array of shape `(m,)` or `(n,)`, in which case it will be broadcast
        along the other axis. Zero means the element is held fixed and one means
        it is free. By default, in orthogonal distance regression mode, `ifixx`
        is set to one for all elements of `x`. In ordinary least squares mode,
        the `x` values are intrinsically fixed.
    delta0 : Float64Array | None
        Array with the same shape as `x`, containing the initial guesses of the
        errors in the explanatory variable. To activate this option, `job` must
        be set accordingly. By default, `delta0` is set to zero for all elements
        of `x`.
    lower : Float64Vector | None
        Rank-1 array with the same shape as `beta0`, containing the lower bounds
        of the model parameters. By default, `lower` is set to negative infinity
        for all elements of `beta`.
    upper : Float64Vector | None
        Rank-1 array with the same shape as `beta0`, containing the upper bounds
        of the model parameters. By default, `upper` is set to positive infinity
        for all elements of `beta`.
    job : Variable controlling problem initialization and computational method.
        The default value is 0, corresponding to an explicit orthogonal distance
        regression, with `delta0` initialized to zero, derivatives computed by
        forward finite difference, and covariance matrix computed using Jacobian
        matrices recomputed at the final solution. Another common option is 20,
        corresponding to an explicit orthogonal distance regression with 
        user-supplied jacobians `fjacb` and `fjacd`. To initialize `delta0` with
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
        parameter (`beta` and `delta`) convergence. When the model is explicit, 
        the default value is `eps**(2/3)`, and when the model is implicit, the
        default value is `eps**(1/3)`, where `eps` is the machine precision in
        `float64`.
    maxit : int | None
        Maximum number of allowed iterations. The default value is 50 for a
        (normal) first run and 10 for a restart (see `job`).
    stpb : Float64Vector | None
        Rank-1 array with the same shape as `beta0` containing the _relative_
        step sizes used to compute the finite difference derivatives with respect
        to the model parameters. By default, `stpb` is set internally based on
        the value of `ndigit` and the type of finite differences used. For
        additional details, refer to pages 31 and 78 of the ODRPACK95 guide.
    stpd : Float64Array | None
        Array with the same shape as `x`, containing the _relative_ step sizes
        used to compute the finite difference derivatives with respect to the
        errors in the explanatory variable. Alternatively, it can be a rank-1
        array of shape `(m,)` or `(n,)`, in which case it will be broadcast along
        the other axis. By default, `stpd` is set internally based on the value
        of `ndigit` and the type of finite differences used. For additional
        details, refer to pages 31 and 78 of the ODRPACK95 guide.
    sclb : Float64Vector | None
        Rank-1 array with the same shape as `beta0` containing the scale values
        of the model parameters. Scaling is used to improve the numerical stability
        of the regression, but does not affect the problem specification. Scaling
        should not be confused with the weighting matrices `we` and `wd`. By
        default, `sclb` is set internally based on the relative magnitudes of 
        `beta`. For further details, refer to page 32 and 84 of the ODRPACK95
        guide.
    scld : Float64Array | None
        Array with the same shape as `x`, containing the scale values of the
        errors in the explanatory variable. Alternatively, it can be a rank-1
        array of shape `(m,)` or `(n,)`, in which case it will be broadcast along
        the other axis. Scaling is used to improve the numerical stability of
        the regression, but does not affect the problem specification. Scaling
        should not be confused with the weighting matrices `we` and `wd`. By
        default, `scld` is set internally based on the relative magnitudes of
        `x`. For further details, refer to page 32 and 85 of the ODRPACK95 guide.
    work : Float64Vector | None
        Array containing the real-valued internal state of the odrpack solver.
        It is only required for a restart (see `job`), in which case it must be
        set to the state of the previous run.
    iwork : Int32Vector | None
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
    """

    # Interpret job
    isodr = _getdigit(job, 1) < 2
    isjac = _getdigit(job, 2) > 1
    isdelta0 = _getdigit(job, 4) > 0
    isrestart = _getdigit(job, 5) > 0

    # Check x and y
    if x.ndim == 1:
        m = 1
    elif x.ndim == 2:
        m = x.shape[0]
    else:
        raise ValueError(
            f"`x` must be a rank-1 array of shape `(n,)` or a rank-2 array of shape `(m, n)`, but has shape {x.shape}.")

    if y.ndim == 1:
        nq = 1
    elif y.ndim == 2:
        nq = y.shape[0]
    else:
        raise ValueError(
            f"`y` must be a rank-1 array of shape `(n,)` or a rank-2 array of shape `(nq, n)`, but has shape {y.shape}.")

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
    if isdelta0 and delta0 is not None:
        if delta0.shape != x.shape:
            raise ValueError("`delta0` must have the same shape as `x`.")
        delta = delta0.copy()
    elif not isdelta0 and delta0 is None:
        delta = np.zeros_like(x)
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
            we = np.full((nq,), we, dtype=np.float64)
        elif isinstance(we, np.ndarray):
            if we.shape == (nq,):
                ldwe = 1
                ld2we = 1
            elif we.shape == (nq, nq):
                ldwe = 1
                ld2we = nq
            elif we.shape == (nq, n) or (we.shape == (n,) and nq == 1):
                ldwe = n
                ld2we = 1
            elif we.shape in ((nq, 1, 1), (nq, 1, n), (nq, nq, 1), (nq, nq, n)):
                ldwe = we.shape[2]
                ld2we = we.shape[1]
            else:
                raise ValueError(
                    r"`we` must be a array of shape `(nq,)`, `(n,)`, `(nq, nq)`, `(nq, n)`, `(nq, 1, 1)`, `(nq, 1, n)`, `(nq, nq, 1)`, or `(nq, nq, n)`. See page 25 of the ODRPACK95 User Guide.")
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
    if f0.shape != y.shape:
        raise ValueError(
            "Function `f` must return an array with the same shape as `y`.")

    def fdummy(beta, x): return np.array([np.nan])  # should never be called

    if isjac and fjacb is not None:
        fjacb0 = fjacb(beta0, x)
        if fjacb0.shape[-1] != n or fjacb0.size != n*npar*nq:
            raise ValueError(
                "Function `fjacb` must return an array with shape `(n, npar, nq)` or compatible.")
    elif not isjac and fjacb is None:
        fjacb = fdummy
    else:
        raise ValueError("Inconsistent arguments for `job` and `fjacb`.")

    if isjac and fjacd is not None:
        fjacd0 = fjacd(beta0, x)
        if fjacd0.shape[-1] != n or fjacd0.size != n*m*nq:
            raise ValueError(
                "Function `fjacd` must return an array with shape `(n, m, nq)` or compatible.")
    elif not isjac and fjacd is None:
        fjacd = fdummy
    else:
        raise ValueError("Inconsistent arguments for `job` and `fjacd`.")

    # Check/allocate work arrays
    lwork, liwork = workspace_dimensions(n, m, npar, nq, isodr)
    if (not isrestart) and (work is None) and (iwork is None):
        work = np.zeros(lwork, dtype=np.float64)
        iwork = np.zeros(liwork, dtype=np.int32)
    elif isrestart and (work is not None) and (iwork is not None):
        if work.size != lwork:
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
    info = _odr(n=n, m=m, npar=npar, nq=nq,
                ldwe=ldwe, ld2we=ld2we,
                ldwd=ldwd, ld2wd=ld2wd,
                ldifx=ldifx,
                ldstpd=ldstpd, ldscld=ldscld,
                f=f, fjacb=fjacb, fjacd=fjacd,
                beta=beta, y=y, x=x,
                delta=delta,
                we=we, wd=wd, ifixb=ifixb, ifixx=ifixx,
                lower=lower, upper=upper,
                work=work, iwork=iwork,
                job=job,
                ndigit=ndigit, taufac=taufac, sstol=sstol, partol=partol, maxit=maxit,
                iprint=iprint, errfile=errfile, rptfile=rptfile
                )

    # Indexes of integer and real work arrays
    iwork_idx: dict[str, int] = diwinf(m, npar, nq)
    work_idx: dict[str, int] = dwinf(n, m, npar, nq, ldwe, ld2we, isodr)

    # Return the result
    # Extract results without messing up the original work arrays
    i0_eps = work_idx['eps']
    eps = work[i0_eps:i0_eps+y.size].copy()
    eps = np.reshape(eps, y.shape)

    i0_sd = work_idx['sd']
    sd_beta = work[i0_sd:i0_sd+beta.size].copy()

    i0_vcv = work_idx['vcv']
    cov_beta = work[i0_vcv:i0_vcv+beta.size**2].copy()
    cov_beta = np.reshape(cov_beta, (beta.size, beta.size))

    result = OdrResult(
        beta=beta,
        delta=delta,
        eps=eps,
        xplus=x+delta,
        yest=y+eps,
        sd_beta=sd_beta,
        cov_beta=cov_beta,
        res_var=work[work_idx['rvar']],
        info=info,
        stopreason=_interpret_info(info),
        success=info < 4,
        nfev=iwork[iwork_idx['nfev']],
        njev=iwork[iwork_idx['njev']],
        niter=iwork[iwork_idx['niter']],
        irank=iwork[iwork_idx['irank']],
        inv_condnum=work[work_idx['rcond']],
        sum_square=work[work_idx['wss']],
        sum_square_delta=work[work_idx['wssde']],
        sum_square_eps=work[work_idx['wssep']],
        iwork=iwork,
        work=work,
    )

    return result


def _getdigit(number: int, ndigit: int) -> int:
    """Return the `ndigit`-th digit from the right of `number`."""
    return (number // 10**(ndigit-1)) % 10


def _interpret_info(info: int) -> str:
    """Return a message corresponding to the value of `info`."""
    message = ""
    if info == 1:
        message = "Sum of squares convergence."
    elif info == 2:
        message = "Parameter convergence."
    elif info == 3:
        message = "Sum of squares and parameter convergence."
    elif info == 4:
        message = "Iteration limit reached."
    elif info >= 5:
        message = "Questionable results or fatal errors detected. See report and error message."
    return message
