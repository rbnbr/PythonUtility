import numpy as np
from scipy.interpolate import RegularGridInterpolator


def fill_array_with(to_be_filled: np.array, data: np.array, offsets=None):
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


def resample_array_to_shape(array: np.array, new_shape, method="linear"):
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


def interp_variables(x: np.array, xp=lambda x, i: (x.min(), x.max()), fp=lambda x, i: (0, 255)):
    """
    Rescales the values of x with respect to each variable of the last axis.

    xp and fp are evaluated for value in range(x.shape[-1]) and called with arguments: (x[:, ..., :, i], i)

    :param x: np.array
    :param xp: func(x: np.array, index: int) -> tuple(number, number)
    :param fp: func(x: np.array, index: int) -> tuple(number, number)
    :return:
    """
    if len(x.shape) > 1:
        x = x.swapaxes(0, -1)

    for i in range(x.shape[0]):
        x[i] = np.interp(x[i], xp=xp(x[i], i), fp=fp(x[i], i))

    x = x.swapaxes(0, -1)
    return x
