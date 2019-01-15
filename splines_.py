import numpy as np
from scipy import linalg
import patsy

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


def get_basis_matrix(timepoints, conditions=None, knots=None,
                     df=4, degree=3, orthonormalize=False,
                     include_intercept=False,
                     splines="bs"):
    """
    Returns the basis matrix

    Parameters
    ----------
    timepoints : ndarray (t, ) of timepoints

    conditions : ndarray (t, ), optional, default: None
        ndarray containing the conditions information.

    knots : ndarray, optional, default: None
        The interior knots to use for the spline. If unspecified, then equally
        spaced quantiles of the input data are used. You must specify at least
        one of ``df`` and ``knots``

    df : int, default 4
        Number of degrees of freedom of the spline. This corresponds to the
        size of the basis

    degree : int, default: 3
        The degree of the splines to use

    orthonormalize : boolean, default: false

    include_intercept : boolean, default: False

    splines : {"bs", "cc", "cr"}
        the types of splines:
            bs: B-splines
            cc: cyclic cubic splines
            cr: natural smoothing splines

    Returns
    -------
    basis : ndarray (t, k)
    """
    # If conditions are not provided, just return the basis for a single
    # condition
    if conditions is None or len(np.unique(conditions)) == 1:
        model = (
            "{}(x, df={}, knots={}) ".format(
                splines,
                df, knots))
        data = {"x": timepoints}
        if not include_intercept:
            model = model + " + 0"
    else:
        model = (
            "C(conditions):{}(x, df={}, knots={}) ".format(
                splines,
                df, knots))

        data = {"x": timepoints,
                "conditions": conditions}
        if include_intercept:
            model = model + " + C(conditions)"

        model = model + " + 0"
    basis = patsy.dmatrix(model, data)
    if orthonormalize:
        basis = linalg.orth(basis)
    return basis
