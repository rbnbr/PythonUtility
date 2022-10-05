import numpy as np
from scipy.spatial.distance import pdist, squareform


def flatten_extra_dimensions(x: np.ndarray):
    """
    Makes arrays one dimensional, but keeps last dimension as 1.
    :param x:
    :return:
    """
    if len(x.shape) <= 0:
        return x
    elif len(x.shape) == 1:
        return x.reshape((x.shape[0], 1))
    else:
        return x.reshape((x.shape[0], int(np.prod(x.shape[1:]))))


def transform_to_rect_flat_array(x: list):
    """
    Returns a numpy array which has rectangular form and the first column contains the index of the row
    and the second column contains the length of the row (minus the first two columns) with actual data.

    The metric_wrapper will work with this kind of data.

    e.g.: 3x7
    0 4 0.1 2 -1 4 NaN NaN NaN
    1 2 -2.1 31 NaN NaN NaN NaN NaN
    2 7 1 2 3 4 5 6 7

    :param x:
    :return:
    """
    offsets = [len(a) for a in x]
    max_len = max(offsets)
    rect = np.empty((len(x), max_len + 2))

    for i, a in enumerate(x):
        rect[i][0] = i
        rect[i][1] = offsets[i]
        rect[i][2:offsets[i]+2] = a[:]

    return rect


def metric_wrapper(shapes: list, metric_fn):
    """
    Scipy pairwise distance expects a feature vector but in my use cases, sometimes the feature is a matrix.
    Thus before calling the scipy pdist function, I flatten the array but then I will need to bring it back in original shape before applying the actual metric.
    This wrapper returns a metric which expects a flattened ndarray and will reshape it back to the given shape before calling the actual metric.

    The returned function expects input from transform_to_rect_flat_array(..).
    """
    def w(n_features_x, n_features_y):
        # reverse transform_to_rect_flat_array changes
        x_i = int(n_features_x[0])
        y_i = int(n_features_y[0])

        n_features_x = n_features_x[2:int(n_features_x[1]) + 2]
        n_features_y = n_features_y[2:int(n_features_y[1]) + 2]

        # reverse flattening
        n_features_x = n_features_x.reshape(shapes[x_i])
        n_features_y = n_features_y.reshape(shapes[y_i])

        return metric_fn(n_features_x, n_features_y)

    return w


def pairwise_distance(ensemble: list, metric=np.linalg.norm, precompute=None):
    """
    Compute pairwise distance between elements in ensemble
    :param ensemble: list of np.array
    :param metric:
    :param precompute:
    :return:
    """
    # perform possible precomputations of the metric to avoid this in during the pairwise distance metric
    if precompute is not None:
        ensemble = list(map(precompute, ensemble))

    # handle scalars
    for i in range(len(ensemble)):
        if type(ensemble[i]) == int or type(ensemble[i]) == float:
            ensemble[i] = np.array([ensemble[i]])

    # get shapes for rebuilding after flattening
    shapes = [e.shape for e in ensemble]

    # flatten all dimensions
    e_flat = [e.flatten() for e in ensemble]

    # make it rectangular to be applicable to pdist
    e_rect = transform_to_rect_flat_array(e_flat)

    # compute pairwise distances
    sim_mat = squareform(pdist(e_rect, metric=metric_wrapper(shapes, metric)))

    return sim_mat
