import numpy as np
from scipy.interpolate import RegularGridInterpolator


def fill_array_with(to_be_filled: np.ndarray, data: np.ndarray, offsets=None):
    """
    Fills the array to_be_filled with the data of the array data starting at provided offsets in to_be_filled
    :param to_be_filled:
    :param data:
    :param offsets:
    :return:
    """

    if offsets is None:
        offsets = np.zeros((len(to_be_filled.shape),), dtype=int)

    selector = tuple([slice(offsets[i], offsets[i] + data.shape[i]) for i in range(len(data.shape))])

    to_be_filled[selector] = data[:]

    return to_be_filled


def resample_array_to_shape(array: np.ndarray, new_shape, method="linear"):
    """
    Resamples the array to the provided shape using RegularGridInterpolator with specified method
    :param array:
    :param new_shape:
    :param method:
    :return:
    """
    # generate data
    entries = [np.arange(s) for s in array.shape]
    interp = RegularGridInterpolator(entries, array, method=method)

    # new grid
    new_entries = [np.linspace(0, array.shape[i] - 1, new_shape[i]) for i in range(len(array.shape))]
    new_grid = np.meshgrid(*new_entries, indexing='ij')

    return interp(tuple(new_grid)).astype(array.dtype)


def interp_variables(x: np.ndarray, xp=lambda x, i: (x.min(), x.max()), fp=lambda x, i: (0, 255), in_place=False):
    """
    Rescales the values of x with respect to each variable of the last axis.

    xp and fp are evaluated for value in range(x.shape[-1]) and called with arguments: (x[:, ..., :, i], i)

    :param in_place: if not in_place, calls x.copy() before altering x.
    :param x: np.ndarray
    :param xp: func(x: np.ndarray, index: int) -> tuple(number, number)
    :param fp: func(x: np.ndarray, index: int) -> tuple(number, number)
    :return:
    """
    if not in_place:
        x = x.copy()

    for i in range(x.shape[-1]):
        x[..., i] = np.interp(x[..., i], xp=xp(x[..., i], i), fp=fp(x[..., i], i))

    return x
