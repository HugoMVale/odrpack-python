from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ['OdrResult']


@dataclass(frozen=True)
class OdrResult():
    """
    Results of an Orthogonal Distance Regression (ODR) computation.

    Attributes
    ----------
    beta : NDArray[np.float64]
        Estimated parameters of the model.
    delta : NDArray[np.float64]
        Differences between the observed and fitted `x` values.
    eps : NDArray[np.float64]
        Differences between the observed and fitted `y` values.
    xplus : NDArray[np.float64]
        Adjusted `x` values after fitting, `x + delta`.
    yest : NDArray[np.float64]
        Estimated `y` values corresponding to the fitted model, `y + eps`.
    sd_beta : NDArray[np.float64]
        Standard deviations of the estimated parameters.
    cov_beta : NDArray[np.float64]
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
    iwork : NDArray[np.int32]
        Integer workspace array used internally by `odrpack`.
    rwork : NDArray[np.float64]
        Floating-point workspace array used internally by `odrpack`.
    """
    beta: NDArray[np.float64]
    delta: NDArray[np.float64]
    eps: NDArray[np.float64]
    xplus: NDArray[np.float64]
    yest: NDArray[np.float64]
    sd_beta: NDArray[np.float64]
    cov_beta: NDArray[np.float64]
    res_var: float
    nfev: int
    njev: int
    niter: int
    irank: int
    inv_condnum: float
    info: int
    success: bool
    sum_square: float
    sum_square_delta: float
    sum_square_eps: float
    iwork: NDArray[np.int32]
    rwork: NDArray[np.float64]

    @property
    def stopreason(self) -> str:
        message = ""
        if self.info == 1:
            message = "Sum of squares convergence."
        elif self.info == 2:
            message = "Parameter convergence."
        elif self.info == 3:
            message = "Sum of squares and parameter convergence."
        elif self.info == 4:
            message = "Iteration limit reached."
        elif self.info >= 5:
            message = "Questionable results or fatal errors detected. See report and error message."
        return message
