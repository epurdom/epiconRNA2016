import numpy as np
from scipy import linalg

"""
Extracting from tritonis elements that are useful
"""


def estimate_splines_coefficient(X, y):
    """
    Estimate the spline coefficients

    Parameters
    ----------
    X : ndarray
        the basis matrix

    y : ndarray
        The actual data to regress…
    """
    # XXX oh gosh… seriously???
    beta = np.dot(np.dot(y, X), linalg.inv(np.dot(X.T, X)))
    return beta
