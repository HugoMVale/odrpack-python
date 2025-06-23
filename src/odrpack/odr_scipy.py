from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from odrpack.__odrpack import loc_iwork, loc_rwork
from odrpack.__odrpack import odr as _odr
from odrpack.__odrpack import workspace_dimensions
from odrpack.result import OdrResult

__all__ = ['odr_fit']


def odr_fit(f: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
            xdata: NDArray[np.float64],
            ydata: NDArray[np.float64],
            beta0: NDArray[np.float64],
            *,
            weight_x: float | NDArray[np.float64] | None = None,
            weight_y: float | NDArray[np.float64] | None = None,
            bounds: tuple[NDArray[np.float64] | None,
                          NDArray[np.float64] | None] | None = None,
            task: Literal['explicit-ODR', 'implicit-ODR', 'OLS'] = 'explicit-ODR',
            fix_beta: NDArray[np.bool] | None = None,
            fix_x: NDArray[np.bool] | None = None,
            jac_beta: Callable[[NDArray[np.float64], NDArray[np.float64]],
                               NDArray[np.float64]] | None = None,
            jac_x: Callable[[NDArray[np.float64], NDArray[np.float64]],
                            NDArray[np.float64]] | None = None,
            delta0: NDArray[np.float64] | None = None,
            diff_scheme: Literal['forward', 'central'] = 'forward',
            report: Literal['none', 'short', 'long', 'iteration'] = 'none',
            maxit: int = 50,
            ndigit: int | None = None,
            taufac: float | None = None,
            sstol: float | None = None,
            partol: float | None = None,
            step_beta: NDArray[np.float64] | None = None,
            step_delta: NDArray[np.float64] | None = None,
            scale_beta: NDArray[np.float64] | None = None,
            scale_delta: NDArray[np.float64] | None = None,
            rptfile: str | None = None,
            errfile: str | None = None,
            ) -> OdrResult:
    r"""Solve a weighted orthogonal distance regression (ODR) problem, also
    known as errors-in-variables regression.

    Parameters
    ----------
    f : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
        Function to be fitted, with the signature `f(beta, x)`. It must return
        an array with the same shape as `y`.
    xdata : NDArray[np.float64]
        Array of shape `(n,)` or `(m, n)` containing the observed values of the
        explanatory variable(s).
    ydata : NDArray[np.float64]
        Array of shape `(n,)` or `(q, n)` containing the observed values of the
        response variable(s). When the model is explicit, `ydata` must contain
        a value for each observation. To ignore specific values (e.g., missing
        data), set the corresponding entry in `weight_y` to zero. When the model
        is implicit, `ydata` is not used (but must be defined).
    beta0 : NDArray[np.float64]
        Array of shape `(npar,)` with the initial guesses of the model parameters,
        within the optional bounds specified by `bounds`.
    weight_x : float | NDArray[np.float64] | None
        Scalar or array specifying how the errors on `xdata` are to be weighted.
        If `weight_x` is a scalar, then it is used for all data points. If
        `weight_x` is an array of shape `(n,)` and `m==1`, then `weight_x[i]`
        represents the weight for `xdata[i]`. If `weight_x` is an array of shape
        `(m)`, then it represents the diagonal of the covariant weighting matrix
        for all data points. If `weight_x` is an array of shape `(m, m)`, then
        it represents the full covariant weighting matrix for all data points.
        If `weight_x` is an array of shape `(m, n)`, then `weight_x[:, i]`
        represents the diagonal of the covariant weighting matrix for `xdata[:, i]`.
        If `weight_x` is an array of shape `(m, m, n)`, then `weight_x[:, :, i]`
        represents the full covariant weighting matrix for `xdata[:, i]`. For a
        comprehensive description of the options, refer to page 26 of the
        ODRPACK95 guide. By default, `weight_x` is set to one for all `xdata`
        points.
    weight_y : float | NDArray[np.float64] | None
        Scalar or array specifying how the errors on `ydata` are to be weighted.
        If `weight_y` is a scalar, then it is used for all data points. If
        `weight_y` is an array of shape `(n,)` and `q==1`, then `weight_y[i]`
        represents the weight for `ydata[i]`. If `weight_y` is an array of shape
        `(q)`, then it represents the diagonal of the covariant weighting matrix
        for all data points. If `weight_y` is an array of shape `(q, q)`, then
        it represents the full covariant weighting matrix for all data points.
        If `weight_y` is an array of shape `(q, n)`, then `weight_y[:, i]`
        represents the diagonal of the covariant weighting matrix for `ydata[:, i]`.
        If `weight_y` is an array of shape `(q, q, n)`, then `weight_y[:, :, i]`
        represents the full covariant weighting matrix for `ydata[:, i]`. For a
        comprehensive description of the options, refer to page 25 of the
        ODRPACK95 guide. By default, `weight_y` is set to one for all `ydata`
        points.
    bounds : tuple[NDArray[np.float64] | None, NDArray[np.float64] | None] | None
        Tuple of arrays with the same shape as `beta0`, specifying the lower and
        upper bounds of the model parameters. The first array contains the lower
        bounds, and the second contains the upper bounds. By default, the bounds
        are set to negative and positive infinity, respectively, for all elements
        of `beta`.
    task : Literal['explicit-ODR', 'implicit-ODR', 'OLS']
        Specifies the regression task to be performed. `'explicit-ODR'` solves
        an orthogonal distance regression problem with an explicit model.
        `'implicit-ODR'` handles models defined implicitly. `'OLS'` performs
        ordinary least squares fitting.
    fix_beta : NDArray[np.bool] | None
        Array with the same shape as `beta0`, specifying which elements of `beta`
        are to be held fixed. `True` means the parameter is fixed; `False` means
        it is adjustable. By default, all elements of `beta` are set to `False`.
    fix_x : NDArray[np.bool] | None
        Array with the same shape as `xdata`, specifying which elements of `xdata`
        are to be held fixed. Alternatively, it can be a rank-1 array of shape
        `(m,)` or `(n,)`, in which case it will be broadcast along the other
        axis. `True` means the element is fixed; `False` means it is adjustable.
        By default, in orthogonal distance regression mode, all elements of 
        `fix_x` are set to `False`. In ordinary least squares mode (`task='OLS'`),
        all `xdata` values are automatically treated as fixed.
    jac_beta : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None
        Jacobian of the function to be fitted with respect to `beta`, with the
        signature `jac_beta(beta, x)`. It must return an array with shape 
        `(n, npar, q)` or a compatible shape. By default, the Jacobian is
        approximated numerically using the finite difference scheme specified
        by `diff_scheme`.
    jac_x : Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None
        Jacobian of the function to be fitted with respect to `x`, with the
        signature `jac_x(beta, x)`. It must return an array with shape 
        `(n, m, q)` or a compatible shape. By default, the Jacobian is approximated
        numerically using the finite difference scheme specified by `diff_scheme`.
    delta0 : NDArray[np.float64] | None
        Array with the same shape as `xdata`, containing the initial guesses of 
        the errors in the explanatory variable. By default, `delta0` is set to
        zero for all elements of `xdata`.
    diff_scheme : Literal['forward', 'central']
        Finite difference scheme used to approximate the Jacobian matrices when
        the user does not provide `jac_beta` and `jac_x`. The default method is
        forward differences. Central differences are generally more accurate but
        require one additional `f` evaluation per partial derivative.
    report : Literal['none', 'short', 'long', 'iteration']
        Specifies the level of computation reporting. `'none'` disables all output.
        `'short'` prints a brief initial and final summary. `'long'` provides a 
        detailed initial and final summary. `'iteration'` outputs information at
        each iteration step in addition to the detailed summaries. This is 
        useful for debugging or monitoring progress.
    maxit : int | None
        Maximum number of allowed iterations.
    ndigit : int | None
        Number of reliable decimal digits in the values computed by the model
        function `f` and its Jacobians `jac_beta`, and `jac_x`. By default,
        the value is numerically determined by evaluating `f`. 
    taufac : float | None
        Factor ranging from 0 to 1 to initialize the trust region radius. The
        default value is 1. Reducing `taufac` may be appropriate if, at the
        first iteration, the computed results for the full Gauss-Newton step
        cause an overflow, or cause `beta` and/or `delta` to leave the region
        of interest. 
    sstol : float | None
        Factor ranging from 0 to 1 specifying the stopping tolerance for the
        sum of the squares convergence. The default value is `eps**(1/2)`,
        where `eps` is the machine precision in `float64`.
    partol : float | None
        Factor ranging from 0 to 1 specifying the stopping tolerance for
        parameter convergence (i.e., `beta` and `delta`). When the model is
        explicit, the default value is `eps**(2/3)`, and when the model is
        implicit, the default value is `eps**(1/3)`, where `eps` is the machine
        precision in `float64`.
    step_beta : NDArray[np.float64] | None
        Array with the same shape as `beta0` containing the _relative_ step
        sizes used to compute the finite difference derivatives with respect
        to the model parameters. By default, the step size is set internally 
        based on the value of `ndigit` and the type of finite differences
        specified by `diff_scheme`. For additional details, refer to pages 31
        and 78 of the ODRPACK95 guide.
    step_delta : NDArray[np.float64] | None
        Array with the same shape as `xdata`, containing the _relative_ step 
        sizes used to compute the finite difference derivatives with respect to
        the errors in the explanatory variable. Alternatively, it can be a rank-1
        array of shape `(m,)` or `(n,)`, in which case it will be broadcast along
        the other axis. By default, step size is set internally based on the value
        of `ndigit` and the type of finite differences specified by `diff_scheme`.
        For additional details, refer to pages 31 and 78 of the ODRPACK95 guide.
    scale_beta : NDArray[np.float64] | None
        Array with the same shape as `beta0` containing the scale values of the
        model parameters. Scaling is used to improve the numerical stability
        of the regression, but does not affect the problem specification. Scaling
        should not be confused with the weighting matrices `weight_x` and 
        `weight_y`. By default, the scale is set internally based on the relative
        magnitudes of `beta`. For further details, refer to pages 32 and 84 of
        the ODRPACK95 guide.
    scale_delta : NDArray[np.float64] | None
        Array with the same shape as `xdata`, containing the scale values of the
        errors in the explanatory variable. Alternatively, it can be a rank-1
        array of shape `(m,)` or `(n,)`, in which case it will be broadcast along
        the other axis. Scaling is used to improve the numerical stability of
        the regression, but does not affect the problem specification. Scaling
        should not be confused with the weighting matrices `weight_x` and 
        `weight_y`. By default, the scale is set internally based on the relative
        magnitudes of `xdata`. For further details, refer to pages 32 and 85 of
        the ODRPACK95 guide.
    rptfile : str | None
        File name for storing the computation reports, as defined by `report`.
        By default, the reports are sent to standard output.
    errfile : str | None
        File name for storing the error reports, as defined by `report`. By
        default, the reports are sent to standard output.

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
    >>> from odrpack import odr_fit
    >>> xdata = np.array([0.982, 1.998, 4.978, 6.01])
    >>> ydata = np.array([2.7, 7.4, 148.0, 403.0])
    >>> beta0 = np.array([2., 0.5])
    >>> bounds = (np.array([0., 0.]), np.array([10., 0.9]))
    >>> def f(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    ...     return beta[0] * np.exp(beta[1]*x)
    >>> sol = odr_fit(f, xdata, ydata, beta0, bounds=bounds)
    >>> sol.beta
    array([1.63336897, 0.9       ])
    """

    # Check xdata and ydata
    if xdata.ndim == 1:
        m = 1
    elif xdata.ndim == 2:
        m = xdata.shape[0]
    else:
        raise ValueError(
            f"`xdata` must be a rank-1 array of shape `(n,)` or a rank-2 array of shape `(m, n)`, but has shape {xdata.shape}.")

    if ydata.ndim == 1:
        q = 1
    elif ydata.ndim == 2:
        q = ydata.shape[0]
    else:
        raise ValueError(
            f"`ydata` must be a rank-1 array of shape `(n,)` or a rank-2 array of shape `(q, n)`, but has shape {ydata.shape}.")

    if xdata.shape[-1] == ydata.shape[-1]:
        n = xdata.shape[-1]
    else:
        raise ValueError(
            f"The last dimension of `xdata` and `ydata` must be identical, but x.shape={xdata.shape} and y.shape={ydata.shape}.")

    # Check beta0
    if beta0.ndim == 1:
        npar = beta0.size
        beta = beta0.copy()
    else:
        raise ValueError(
            f"`beta0` must be a rank-1 array of shape `(npar,)`, but has shape {beta0.shape}.")

    # Check p bounds
    if bounds is not None:
        lower, upper = bounds
        if lower is not None:
            if lower.shape != beta0.shape:
                raise ValueError(
                    "The lower bound `bounds[0]` must have the same shape as `beta0`.")
            if np.any(lower >= beta0):
                raise ValueError(
                    "The lower bound `bounds[0]` must be less than `beta0`.")
        if upper is not None:
            if upper.shape != beta0.shape:
                raise ValueError(
                    "The upper bound `bounds[1]` must have the same shape as `beta0`.")
            if np.any(upper <= beta0):
                raise ValueError(
                    "The upper bound `bounds[1]` must be greater than `beta0`.")
    else:
        lower, upper = None, None

    # Check other beta related arguments
    if fix_beta is not None and fix_beta.shape != beta0.shape:
        raise ValueError("`fix_beta` must have the same shape as `beta0`.")

    if step_beta is not None and step_beta.shape != beta0.shape:
        raise ValueError("`step_beta` must have the same shape as `beta0`.")

    if scale_beta is not None and scale_beta.shape != beta0.shape:
        raise ValueError("`scale_beta` must have the same shape as `beta0`.")

    # Check delta0
    if delta0 is not None:
        if delta0.shape != xdata.shape:
            raise ValueError("`delta0` must have the same shape as `xdata`.")
        delta = delta0.copy()
        has_delta0 = True
    else:
        delta = np.zeros(xdata.shape, dtype=np.float64)
        has_delta0 = False

    # Check fix_x
    if fix_x is not None:
        if fix_x.shape == xdata.shape:
            ldifx = n
        elif fix_x.shape == (m,) and m > 1 and n != m:
            ldifx = 1
        elif fix_x.shape == (n,) and m > 1 and n != m:
            ldifx = n
            fix_x = np.tile(fix_x, (m, 1))
        else:
            raise ValueError(
                "`fix_x` must either have the same shape as `xdata` or be a rank-1 array of shape `(m,)` or `(n,)`. See page 26 of the ODRPACK95 User Guide.")
    else:
        ldifx = 1

    # Check step_delta
    if step_delta is not None:
        if step_delta.shape == xdata.shape:
            ldstpd = n
        elif step_delta.shape == (m,) and m > 1 and n != m:
            ldstpd = 1
        elif step_delta.shape == (n,) and m > 1 and n != m:
            ldstpd = n
            step_delta = np.tile(step_delta, (m, 1))
        else:
            raise ValueError(
                "`step_delta` must either have the same shape as `xdata` or be a rank-1 array of shape `(m,)` or `(n,)`. See page 31 of the ODRPACK95 User Guide.")
    else:
        ldstpd = 1

    # Check scale_delta
    if scale_delta is not None:
        if scale_delta.shape == xdata.shape:
            ldscld = n
        elif scale_delta.shape == (m,) and m > 1 and n != m:
            ldscld = 1
        elif scale_delta.shape == (n,) and m > 1 and n != m:
            ldscld = n
            scale_delta = np.tile(scale_delta, (m, 1))
        else:
            raise ValueError(
                "`scale_delta` must either have the same shape as `xdata` or be a rank-1 array of shape `(m,)` or `(n,)`. See page 32 of the ODRPACK95 User Guide.")
    else:
        ldscld = 1

    # Check weight_x
    if weight_x is not None:
        if isinstance(weight_x, (float, int)):
            ldwd = 1
            ld2wd = 1
            weight_x = np.full((m,), weight_x, dtype=np.float64)
        elif isinstance(weight_x, np.ndarray):
            if weight_x.shape == (m,):
                ldwd = 1
                ld2wd = 1
            elif weight_x.shape == (m, m):
                ldwd = 1
                ld2wd = m
            elif weight_x.shape == (m, n) or (weight_x.shape == (n,) and m == 1):
                ldwd = n
                ld2wd = 1
            elif weight_x.shape in ((m, 1, 1), (m, 1, n), (m, m, 1), (m, m, n)):
                ldwd = weight_x.shape[2]
                ld2wd = weight_x.shape[1]
            else:
                raise ValueError(
                    r"`weight_x` must be a array of shape `(m,)`, `(n,)`, `(m, m)`, `(m, n)`, `(m, 1, 1)`, `(m, 1, n)`, `(m, m, 1)`, or `(m, m, n)`. See page 26 of the ODRPACK95 User Guide.")
        else:
            raise TypeError("`weight_x` must be a float or an array.")
    else:
        ldwd = 1
        ld2wd = 1

    # Check weight_y
    if weight_y is not None:
        if isinstance(weight_y, (float, int)):
            ldwe = 1
            ld2we = 1
            weight_y = np.full((q,), weight_y, dtype=np.float64)
        elif isinstance(weight_y, np.ndarray):
            if weight_y.shape == (q,):
                ldwe = 1
                ld2we = 1
            elif weight_y.shape == (q, q):
                ldwe = 1
                ld2we = q
            elif weight_y.shape == (q, n) or (weight_y.shape == (n,) and q == 1):
                ldwe = n
                ld2we = 1
            elif weight_y.shape in ((q, 1, 1), (q, 1, n), (q, q, 1), (q, q, n)):
                ldwe = weight_y.shape[2]
                ld2we = weight_y.shape[1]
            else:
                raise ValueError(
                    r"`weight_y` must be a array of shape `(q,)`, `(n,)`, `(q, q)`, `(q, n)`, `(q, 1, 1)`, `(q, 1, n)`, `(q, q, 1)`, or `(q, q, n)`. See page 25 of the ODRPACK95 User Guide.")
        else:
            raise TypeError("`weight_y` must be a float or an array.")
    else:
        ldwe = 1
        ld2we = 1

    # Check model function
    f0 = f(beta0, xdata)
    if f0.shape != ydata.shape:
        raise ValueError(
            "Function `f` must return an array with the same shape as `ydata`.")

    # Check model jacobians
    def fdummy(beta, x): return np.array([np.nan])  # will never be called

    if jac_beta is None and jac_x is None:
        has_jac = False
        jac_beta = fdummy
        jac_x = fdummy
    elif jac_beta is not None and jac_x is not None:
        has_jac = True
        jac0_beta = jac_beta(beta0, xdata)
        if jac0_beta.shape[-1] != n or jac0_beta.size != n*npar*q:
            raise ValueError(
                "Function `jac_beta` must return an array with shape `(n, npar, q)` or compatible.")
        jac0_x = jac_x(beta0, xdata)
        if jac0_x.shape[-1] != n or jac0_x.size != n*m*q:
            raise ValueError(
                "Function `jac_x` must return an array with shape `(n, m, q)` or compatible.")
    else:
        raise ValueError(
            "Inconsistent arguments for `jac_beta` and `jac_x`. Either both must be provided or none.")

    # Set iprint flag
    iprint_mapping = {
        'none': 0,
        'short': 1001,
        'long': 2002,
        'iteration': 2212
    }
    iprint = iprint_mapping[report]

    # Set job flag
    jobl = [0, 0, 0, 0, 0]

    if task == "explicit-ODR":
        jobl[-1] = 0
        is_odr = True
    elif task == "implicit-ODR":
        jobl[-1] = 1
        is_odr = True
    elif task == "OLS":
        jobl[-1] = 2
        is_odr = False
    else:
        raise ValueError(
            f"Invalid value for `task`: {task}.")

    if has_jac:
        jobl[-2] = 2
    else:
        if diff_scheme == "forward":
            jobl[-2] = 0
        elif diff_scheme == "central":
            jobl[-2] = 1
        else:
            raise ValueError(
                f"Invalid value for `diff_scheme`: {diff_scheme}.")

    if has_delta0:
        jobl[-4] = 1

    job = sum(jobl[i] * 10 ** (len(jobl) - 1 - i) for i in range(len(jobl)))

    # Allocate work arrays (drop restart possibility)
    lrwork, liwork = workspace_dimensions(n, m, q, npar, is_odr)
    rwork = np.zeros(lrwork, dtype=np.float64)
    iwork = np.zeros(liwork, dtype=np.int32)

    # Convert fix to ifix
    ifixb = (~fix_beta).astype(np.int32) if fix_beta is not None else None
    ifixx = (~fix_x).astype(np.int32) if fix_x is not None else None

    # Call the ODRPACK95 routine
    # Note: beta, delta, work, and iwork are modified in place
    info = _odr(
        n=n, m=m, q=q, npar=npar,
        ldwe=ldwe, ld2we=ld2we,
        ldwd=ldwd, ld2wd=ld2wd,
        ldifx=ldifx,
        ldstpd=ldstpd, ldscld=ldscld,
        f=f, fjacb=jac_beta, fjacd=jac_x,
        beta=beta, y=ydata, x=xdata,
        delta=delta,
        we=weight_y, wd=weight_x, ifixb=ifixb, ifixx=ifixx,
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
    eps = rwork[i0_eps:i0_eps+ydata.size].copy()
    eps = np.reshape(eps, ydata.shape)

    i0_sd = rwork_idx['sd']
    sd_beta = rwork[i0_sd:i0_sd+beta.size].copy()

    i0_vcv = rwork_idx['vcv']
    cov_beta = rwork[i0_vcv:i0_vcv+beta.size**2].copy()
    cov_beta = np.reshape(cov_beta, (beta.size, beta.size))

    result = OdrResult(
        beta=beta,
        delta=delta,
        eps=eps,
        xplus=xdata+delta,
        yest=ydata+eps,
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
        sum_square_delta=rwork[rwork_idx['wssde']],
        sum_square_eps=rwork[rwork_idx['wssep']],
        iwork=iwork,
        rwork=rwork,
    )

    return result
