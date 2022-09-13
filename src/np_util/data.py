import numpy as np


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
