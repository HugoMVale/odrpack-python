__all__ = ['OdrStop']


class OdrStop(Exception):
    """
    Exception to stop the fitting process.

    This exception can be raised in the model function or its Jacobians to
    instruct the `odr` routine to stop fitting.
    """
    pass
