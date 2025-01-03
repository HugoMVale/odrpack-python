from __future__ import annotations
import numpy
import typing
__all__ = ['diwinf', 'dwinf', 'odr', 'workspace_dimensions']
def diwinf(m: int, npar: int, nq: int) -> dict:
    """
    Get storage locations within the integer work space.
    
    Parameters
    ----------
    m : int
        Number of columns of data in the explanatory variable.
    npar : int
        Number of function parameters.
    nq : int
        Number of responses per observation.
    
    Returns
    -------
    dict
        A dictionary containing the 0-based indexes of the integer work array.
    """
def dwinf(n: int, m: int, npar: int, nq: int, ldwe: int, ld2we: int, isodr: bool) -> dict:
    """
    Get storage locations within the real work space.
    
    Parameters
    ----------
    n : int
        Number of observations.
    m : int
        Number of columns of data in the explanatory variable.
    npar : int
        Number of function parameters.
    nq : int
        Number of responses per observation.
    ldwe : int
        Leading dimension of the `we` array.
    ld2we : int
        Second dimension of the `we` array.
    isodr : bool
        Indicates whether the solution is by ODR (True) or by OLS (False).
    
    Returns
    -------
    dict
        A dictionary containing the 0-based indexes of the real work array.
    """
def odr(n: int, m: int, npar: int, nq: int, ldwe: int, ld2we: int, ldwd: int, ld2wd: int, ldifx: int, ldstpd: int, ldscld: int, f: typing.Callable, fjacb: typing.Callable, fjacd: typing.Callable, beta: numpy.ndarray[numpy.float64], y: numpy.ndarray[numpy.float64], x: numpy.ndarray[numpy.float64], delta: numpy.ndarray[numpy.float64], we: numpy.ndarray[numpy.float64] | None = None, wd: numpy.ndarray[numpy.float64] | None = None, ifixb: numpy.ndarray[numpy.int32] | None = None, ifixx: numpy.ndarray[numpy.int32] | None = None, stpb: numpy.ndarray[numpy.float64] | None = None, stpd: numpy.ndarray[numpy.float64] | None = None, sclb: numpy.ndarray[numpy.float64] | None = None, scld: numpy.ndarray[numpy.float64] | None = None, lower: numpy.ndarray[numpy.float64] | None = None, upper: numpy.ndarray[numpy.float64] | None = None, work: numpy.ndarray[numpy.float64] | None = None, iwork: numpy.ndarray[numpy.int32] | None = None, job: int | None = None, ndigit: int | None = None, taufac: float | None = None, sstol: float | None = None, partol: float | None = None, maxit: int | None = None, iprint: int | None = None, errfile: str | None = None, rptfile: str | None = None) -> int:
    """
    C++ wrapper for the Orthogonal Distance Regression (ODR) routine.
    
    Parameters
    ----------
    n : int
        Number of observations.
    m : int
        Number of columns in the independent variable data.
    npar : int
        Number of function parameters.
    nq : int
        Number of responses per observation.
    ldwe : int
        Leading dimension of the `we` array, must be in `{1, n}`.
    ld2we : int
        Second dimension of the `we` array, must be in `{1, nq}`.
    ldwd : int
        Leading dimension of the `wd` array, must be in `{1, n}`.
    ld2wd : int
        Second dimension of the `wd` array, must be in `{1, m}`.
    ldifx : int
        Leading dimension of the `ifixx` array, must be in `{1, n}`.
    ldstpd : int
        Leading dimension of the `stpd` array, must be in `{1, n}`.
    ldscld : int
        Leading dimension of the `scld` array, must be in `{1, n}`.
    f : Callable
        User-supplied function for evaluating the model, `f(beta, x)`.
    fjacb : Callable
        User-supplied function for evaluating the Jacobian w.r.t. `beta`,
        `fjacb(beta, x)`.
    fjacd : Callable
        User-supplied function for evaluating the Jacobian w.r.t. `delta`,
        `fjacd(beta, x)`.
    beta : np.ndarray[float64]
        Array of function parameters with shape `(npar)`.
    y : np.ndarray[float64]
        Dependent variables with shape `(nq, n)`. Ignored for implicit models.
    x : np.ndarray[float64]
        Explanatory variables with shape `(m, n)`.
    delta : np.ndarray[float64]
        Initial errors in `x` data with shape `(m, n)`.
    we : np.ndarray[float64], optional
        Weights for `epsilon` with shape `(nq, ld2we, ldwe)`. Default is None.
    wd : np.ndarray[float64], optional
        Weights for `delta` with shape `(m, ld2wd, ldwd)`. Default is None.
    ifixb : np.ndarray[int32], optional
        Indicates fixed elements of `beta`. Default is None.
    ifixx : np.ndarray[int32], optional
        Indicates fixed elements of `x`. Default is None.
    stpb : np.ndarray[float64], optional
        Relative steps for finite difference derivatives w.r.t. `beta`. Default is None.
    stpd : np.ndarray[float64], optional
        Relative steps for finite difference derivatives w.r.t. `delta`. Default is None.
    sclb : np.ndarray[float64], optional
        Scaling values for `beta`. Default is None.
    scld : np.ndarray[float64], optional
        Scaling values for `delta`. Default is None.
    lower : np.ndarray[float64], optional
        Lower bounds for `beta`. Default is None.
    upper : np.ndarray[float64], optional
        Upper bounds for `beta`. Default is None.
    work : np.ndarray[float64], optional
        Real work space. Default is None.
    iwork : np.ndarray[int32], optional
        Integer work space. Default is None.
    job : int, optional
        Controls initialization and computational method. Default is None.
    ndigit : int, optional
        Number of accurate digits in function results. Default is None.
    taufac : float, optional
        Factor for initial trust region diameter. Default is None.
    sstol : float, optional
        Sum-of-squares convergence tolerance. Default is None.
    partol : float, optional
        Parameter convergence tolerance. Default is None.
    maxit : int, optional
        Maximum number of iterations. Default is None.
    iprint : int, optional
        Print control variable. Default is None.
    errfile : str, optional
        Filename to use for error messages. Default is None.
    rptfile : str, optional
        Filename to use for computation reports. Default is None.
    
    Returns
    -------
    result : dict
        Dictionary with the following keys:
        - beta : np.ndarray[float64]
            Function parameters.
        - delta : np.ndarray[float64]
            Errors in `x` data.
        - work : np.ndarray[float64]
            Real work space.
        - iwork : np.ndarray[int32]
            Integer work space.
        - info : int
            Reason for stopping.
    
    Notes
    -----
    - Ensure all array dimensions and functions are consistent with the provided arguments.
    - Input arrays will automatically be made contiguous and cast to the correct type if necessary.
    """
def workspace_dimensions(n: int, m: int, npar: int, nq: int, isodr: bool) -> tuple[int, int]:
    """
    Calculate the dimensions of the workspace arrays.
    
    Parameters
    ----------
    n : int
        Number of observations.
    m : int
        Number of columns of data in the explanatory variable.
    npar : int
        Number of function parameters.
    nq : int
        Number of responses per observation.
    isodr : bool
        Variable designating whether the solution is by ODR (`True`) or by OLS (`False`).
    
    Returns
    -------
    tuple
        A tuple containing the lengths of the work arrays (`lwork`, `liwork`).
    """
