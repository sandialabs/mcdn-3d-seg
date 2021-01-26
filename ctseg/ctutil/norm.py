"""norm.py
Normalization & Preprocessing Tools

> Tyler Ganter, tganter@sandia.gov
"""
import logging

import numpy as np


logger = logging.getLogger(__name__)


def standardize(x, axis=None):
    """Transforms data to have mean 0 and std dev 1.

    Args:
        x (array-like): the data to standardize
        axis : None or int or tuple of ints, optional
            Axis or axes along which the mean and std are computed. The default is to
            compute the mean and std of the flattened array.

            If this is a tuple of ints, mean + std is performed over multiple axes,
            instead of a single axis or all the axes as before.

    Returns:
        (array-like): standardized version of x
    """
    return (x - np.mean(x, axis=axis)) / np.std(x, axis=axis)
