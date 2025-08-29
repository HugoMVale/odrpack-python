from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ['OdrResult']

F64Array = NDArray[np.float64]
I32Array = NDArray[np.int32]
BoolArray = NDArray[bool]


@dataclass(frozen=True)
class OdrResult():
    """
    Results of an Orthogonal Distance Regression (ODR) computation.

    Attributes
    ----------
    beta : F64Array
        Estimated parameters of the model.
    delta : F64Array
        Differences between the observed and fitted `x` values.
    eps : F64Array
        Differences between the observed and fitted `y` values.
    xplusd : F64Array
        Adjusted `x` values after fitting, `x + delta`.
    yest : F64Array
        Estimated `y` values corresponding to the fitted model, `y + eps`.
    sd_beta : F64Array
        Standard deviations of the estimated parameters.
    cov_beta : F64Array
        Covariance matrix of the estimated parameters.
    res_var : float
        Residual variance, indicating the variance of the residuals.
    nfev : int
        Number of function evaluations during the fitting process.
    njev : int
        Number of Jacobian evaluations during the fitting process.
    niter : int
        Number of iterations performed in the optimization process.
    irank : int
        Rank of the Jacobian matrix at the solution.
    inv_condnum : float
        Inverse of the condition number of the Jacobian matrix.
    info : int
        Status code of the fitting process (e.g., success or failure).
    stopreason : str
        Message indicating the reason for stopping.
    success : bool      
        Whether the fitting process was successful.
    sum_square : float
        Sum of squared residuals (including both `delta` and `eps`).
    sum_square_delta : float
        Sum of squared differences between observed and fitted `x` values.
    sum_square_eps : float
        Sum of squared differences between observed and fitted `y` values.
    iwork : I32Array
        Integer workspace array used internally by `odrpack`. Typically for
        advanced debugging.
    rwork : F64Array
        Floating-point workspace array used internally by `odrpack`. Typically
        for advanced debugging.
    """
    beta: F64Array
    delta: F64Array
    eps: F64Array
    xplusd: F64Array
    yest: F64Array
    sd_beta: F64Array
    cov_beta: F64Array
    res_var: float
    nfev: int
    njev: int
    niter: int
    irank: int
    inv_condnum: float
    info: int
    stopreason: str
    success: bool
    sum_square: float
    sum_square_delta: float
    sum_square_eps: float
    iwork: I32Array
    rwork: F64Array
