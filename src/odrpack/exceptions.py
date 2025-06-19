__all__ = ['OdrStop']


class OdrStop(Exception):
    """
    Exception to stop the regression.

    This exception can be raised in the model function or its Jacobians to
    stop the regression process.
    """
    pass
