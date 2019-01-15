import numpy as np
import itertools
import patsy
from scipy import linalg
import splines_
import warnings


def get_splines_basis_separate(timepoints, conditions=None, df=6, knots=None,
                               include_intercept=False, orthonormalize=False,
                               degree=3,
                               splines="bs"):
    """
    basis : ndarray (t, k)
    """
    mask = timepoints < 8.5
    # If conditions are not provided, just return the basis for a single
    # condition
    if conditions is None or len(np.unique(conditions)) == 1:
        model_equation = ("{}(x, df={}, degree={}, knots={}):(x < 8.5) +"
                          " {}(x, df={}, degree={}, knots={}):(x > 8.5)")
        model = (model_equation.format(
                splines,
                df, degree, knots,
                splines, df, degree, knots))
        data = {"x": timepoints}
        if not include_intercept:
            model = model + " + 0"
    else:
        model_equation = ("C(conditions):{}(x, df={}, degree={}, "
                          "knots={}):(x < 8.5) + C(conditions):{}(x,"
                          " df={}, degree={}, knots={}):(x > 8.5)")
        model = (model_equation.format(
                splines,
                df, degree, knots,
                splines, df, degree, knots))

        data = {"x": timepoints,
                "conditions": conditions}
        if include_intercept:
            model = model + " + C(conditions)"

        model = model + " + 0"
    basis = patsy.dmatrix(model, data)
    basis[mask] = 0

    if orthonormalize:
        basis = linalg.orth(basis)
    return basis


def get_splines_basis(timepoints, conditions=None, df=6, knots=None,
                      include_intercept=False, orthonormalize=False,
                      degree=3,
                      splines="bs"):
    """
    basis : ndarray (t, k)
    """
    # If conditions are not provided, just return the basis for a single
    # condition
    if conditions is None or len(np.unique(conditions)) == 1:
        model = (
            "{}(x, df={}, degree={}, knots={})".format(
                splines,
                df, degree, knots))
        data = {"x": timepoints}
        if not include_intercept:
            model = model + " + 0"
    else:
        model = (
            "C(conditions):{}(x, df={}, degree={}, knots={})".format(
                splines,
                df, degree, knots))

        data = {"x": timepoints,
                "conditions": conditions}
        if include_intercept:
            model = model + " + C(conditions)"

        model = model + " + 0"
    basis = patsy.dmatrix(model, data)
    flowering_timepoints = 8.5
    pre_max = timepoints[basis.argmax(axis=0)] < flowering_timepoints
    mask = (timepoints > flowering_timepoints)[:, np.newaxis] * pre_max
    basis[mask] = 0

    mask = (
        timepoints < flowering_timepoints)[:, np.newaxis] * np.invert(pre_max)
    basis[mask] = 0
    basis[:, 0] = 1

    if orthonormalize:
        basis = linalg.orth(basis)
    return basis


def estimate_shift_and_scaling(centroid, x):
    """
    Estimate the shift and scaling factor of an observation onto the centroid.

    Parameters
    ----------
    centroid : ndarray (p, )

    x : ndarray (n, p)

    Returns
    -------
    scaling, shift
    """
    centroid = centroid.reshape(1, -1)
    if centroid.sum() == 0:
        scaling = 0
        shift = x.mean()
    else:
        scaling = (
            (centroid * x -
             centroid * x.mean(axis=1)[:, np.newaxis]).sum(axis=1) /
            (centroid**2 - centroid*centroid.mean()).sum())

        scaling[scaling < 0] = 0
        shift = ((x - scaling[:, np.newaxis] * centroid)).mean(axis=1)
    return scaling, shift


def align(centroid, x, return_centroids=True):
    """
    Align the centroid on the data x
    """
    centroid = centroid.reshape(1, -1)
    if centroid.sum() == 0:
        a = 0
        b = x.mean()
    else:
        a = ((centroid * x -
              centroid * x.mean(axis=1)[:, np.newaxis]).sum(axis=1) /
             (centroid**2 - centroid*centroid.mean()).sum())

        a[a < 0] = 0
        b = ((x - a[:, np.newaxis] * centroid)).mean(axis=1)

    if return_centroids:
        return (a[:, np.newaxis] * centroid + b[:, np.newaxis])
    else:
        aligned_x = (x - b[:, np.newaxis])
        aligned_x[a != 0] /= a[a != 0][:, np.newaxis]
        if np.any(np.isnan(aligned_x) & np.isinf(aligned_x)):
            raise ValueError("NaN and infs in aligned x")
        return aligned_x


def align_data_onto_centroid(centroid, x):
    """
    Align the data onto the centroid
    """
    centroid = centroid.reshape(1, -1)
    a = (((centroid - centroid.mean()) * x).sum(axis=1) /
         ((x - x.mean(axis=1)[:, np.newaxis])*x).sum(axis=1))

    a[np.isnan(a)] = 0
    a[a < 0] = 0
    centroid = centroid.flatten()
    b = (centroid - a[:, np.newaxis] * x).mean(axis=1)
    return a[:, np.newaxis] * x + b[:, np.newaxis]


def _compute_statistics(data, timepoints, conditions):
    """
    Averages all replicates with one another

    Parameters
    ----------
    data : ndarray [n, p]

    timepoints : ndarray [p, ]

    conditions : ndarray [p, ]
    """
    new_conditions = []
    new_timepoints = []
    mean = []
    variance = []
    # We are also going to keep track of the number of elements per timepoints
    # x conditions.
    num_replicates = []

    for t, c in itertools.product(np.unique(timepoints),
                                  np.unique(conditions)):
        mask = (conditions == c) & (t == timepoints)
        if mask.sum():
            new_conditions.append(c)
            new_timepoints.append(t)
            mean.append(data[:, mask].mean(axis=1)[np.newaxis])
            variance.append(data[:, mask].var(axis=1)[np.newaxis])
            num_replicates.append(mask.sum())

    mean = np.concatenate(mean).T
    variance = np.concatenate(variance).T
    num_replicates = np.array(num_replicates)

    return (mean, variance, num_replicates,
            np.array(new_timepoints), np.array(new_conditions))


def _average_replicates(data, timepoints, conditions):
    """
    Averages all replicates with one another

    Parameters
    ----------
    data : ndarray [n, p]

    timepoints : ndarray [p, ]

    conditions : ndarray [p, ]
    """
    data = data.copy()

    for t, c in itertools.product(np.unique(timepoints),
                                  np.unique(conditions)):
        mask = (conditions == c) & (t == timepoints)
        if mask.sum():
            data[:, mask] = data[:, mask].mean(axis=1)[:, np.newaxis]
    return data


def compute_effect_sizes_per_timepoints(data, timepoints, conditions,
                                        conditions_on_which, standardize=True):
    """
    Compute effect size for each timepoints

    """
    if len(conditions_on_which) > 2:
        raise ValueError("Too many conditions")

    which = np.isin(conditions, conditions_on_which)
    data = data[:, which]
    conditions = conditions[which]
    timepoints = timepoints[which]

    mean = _average_replicates(data, timepoints, conditions)

    effect_sizes = np.zeros((data.shape[0], len(np.unique(timepoints))))
    labels = []

    condition_mask = conditions == conditions_on_which[0]
    if standardize:
        to_prepend = ""
    else:
        to_prepend = "lfc-"
    for i, t in enumerate(np.unique(timepoints)):
        mask = timepoints == t
        # First, check that we really have two conditions for this timepoint
        if len(np.unique(conditions[mask])) == 1:
            effect_sizes[i] = np.nan
            labels.append("%s%s-%s-week%d" % (
                to_prepend,
                conditions_on_which[0],
                conditions_on_which[1],
                t))

            continue

        diff_mean = (np.nanmean(mean[:, mask & condition_mask], axis=1) -
                     np.nanmean(mean[:, mask & ~condition_mask], axis=1))
        variance = np.nanmean((data[:, mask] - mean[:, mask])**2, axis=1)
        if np.any(variance == 0):
            m = variance == 0
            variance[m] = np.nanmean((data - mean) ** 2, axis=1)[m]
        if standardize:
            effect_sizes[:, i] = diff_mean / np.sqrt(variance)
        else:
            effect_sizes[:, i] = diff_mean
        labels.append("%s%s-%s-week%d" % (
            to_prepend,
            conditions_on_which[0],
            conditions_on_which[1],
            t))

    return effect_sizes, labels


def compute_log_fold_change(data, timepoints, conditions, conditions_on_which,
                            squared=True):
    data, _, _, timepoints, conditions = _compute_statistics(
        data, timepoints, conditions)

    if len(conditions_on_which) > 2:
        raise ValueError("Too many conditions")

    conditions = np.array(conditions)
    which = np.isin(conditions, conditions_on_which)
    data = data[:, which]
    conditions = conditions[which]
    timepoints = timepoints[which]

    # At this point, we should have one element of each timepoint. If it
    # doesn't exists, it means teh timepoints is not present in the other
    # condition so remove it

    t, count = np.unique(timepoints, return_counts=True)
    mask = np.isin(timepoints, t[count == 2])
    data = data[:, mask]
    conditions = conditions[mask]
    timepoints = timepoints[mask]

    # Now, we just have to loop over the remaining timepoints and take the
    # squared differences.

    lfc = np.zeros(data.shape[0])
    if squared:
        coef = 2
    else:
        coef = 1

    sign = 0
    for t in np.unique(timepoints):
        mask = t == timepoints
        sign += np.sign(np.diff(data[:, mask], axis=1)).flatten()
        lfc += (np.diff(data[:, mask], axis=1)**coef).flatten()

    return lfc * sign


def compute_effect_size(data, timepoints, conditions, conditions_on_which):
    """
    Compute effect size

    Parameters
    ----------
    """
    # FIXME refactor this function with the one above.

    effect_sizes_ind, labels = compute_effect_sizes_per_timepoints(
        data, timepoints, conditions, conditions_on_which)
    condition = conditions_on_which[0]
    mask = [l.find(condition) != -1 for l in labels]
    effect_size = np.nanmean(effect_sizes_ind[:, mask], axis=1)

    if np.any(np.isnan(effect_size) | np.isinf(effect_size)):
        warnings.warn("NaN in effect size. Replacing with 0")
        effect_size[np.isnan(effect_size) | np.isinf(effect_size)] = 0

    return effect_size


def find_BH_threshold(X, alpha=0.05):
    sv = np.sort(X[np.invert(np.isnan(X))])
    n_tests = len(sv)
    if n_tests == 0:
        return 0
    selected = sv[sv <= float(alpha) / n_tests * np.arange(1, n_tests + 1)]
    if len(selected) == 0:
        return 0
    return selected.max()


def _complicated_spline_fit(X, timepoints, conditions, df=3,
                            data_points_to_use=None,
                            condition="Preflowering"):
    X_fitted = X.copy()
    if condition == "Preflowering":
        pre_timepoints = timepoints < 8.5
    elif condition == "Postflowering":
        pre_timepoints = timepoints < 9.5

    if data_points_to_use is None:
        data_points_to_use = np.array(
            [True for i in range(len(conditions))])

    conditions[pre_timepoints] = [
        "%s.pre" % c for c in conditions[pre_timepoints]]
    for c in np.unique(conditions):
        mask = conditions == c
        t_masked = timepoints[mask]
        basis = np.array(
            get_splines_basis(
                t_masked,
                df=df,
                include_intercept=True))
        coef = splines_.estimate_splines_coefficient(basis, X[:, mask])
        regressed_x = np.dot(coef, basis.T)
        X_fitted[:, mask] = regressed_x
    # For postflowering, we want to fit on everything, but cluster on the
    # "Real data points" only.
    return X_fitted[:, data_points_to_use]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Just basic test
    x = np.random.random((1, 16))
    b = -5
    a = 10
    centroid = (x - b) / a
    centroid = centroid.flatten()
    aligned_centroid = align(centroid, x)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(aligned_centroid)
    ax.plot(x.flatten(), marker=".", linewidth=0)

    aligned_x = align(centroid, x, return_centroids=False)
    ax = axes[1]
    ax.plot(centroid)
    ax.plot(aligned_x.flatten(), marker=".", linewidth=0)
