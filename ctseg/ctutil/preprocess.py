"""preprocess.py
Data Preprocessing Functions

> Tyler Ganter, tganter@sandia.gov
"""
import logging

import numpy as np


logger = logging.getLogger(__name__)


def trim_array(x):
    """Trims the edges of an array (in all dims) that have the same value for the entire
    slice.

    Args:
        x (array-like): the array to trim

    Returns:
        (array-like) a trimmed version of x with shape <= x.shape
    """
    slc = [slice(None)] * x.ndim

    for axis in range(x.ndim):
        lo, hi = _get_same_val_boundaries(x, axis)
        slc[axis] = slice(lo, hi)
        trim_count = x.shape[axis] - hi + 1 + lo
        logger.info(f"Trimming {trim_count}/{x.shape[axis]} rows of axis {axis}")
        x = x[tuple(slc)]

    return x


def _get_same_val_boundaries(x, axis):
    slc = [slice(None)] * x.ndim

    lo_idx = 0
    hi_idx = x.shape[axis]

    for lo_idx in range(x.shape[axis]):
        slc[axis] = lo_idx
        cur_x = x[tuple(slc)]
        if np.any(cur_x != cur_x.item(0)):
            break

    for hi_idx in range(x.shape[axis], 0, -1):
        slc[axis] = hi_idx - 1
        cur_x = x[tuple(slc)]
        if np.any(cur_x != cur_x.item(0)):
            break

    return lo_idx, hi_idx
