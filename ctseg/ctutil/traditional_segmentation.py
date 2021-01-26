"""traditional_segmentation.py
Traditional Non-learned Segmentation Methods

> Tyler Ganter, tganter@sandia.gov
"""
import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def apply_naive_threshold(x, threshold=0.5):
    """Apply naive thresholding to an array.

    The threshold is scaled to the data range before being applied.

    Args:
        x (array-like): the input array
        threshold: threshold in range [0, 1]

    Returns:
        a boolean array of the same dims as `x`
    """
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    scaled_threshold = x_min + (x_max - x_min) * threshold
    return x > scaled_threshold


def erode_and_dilate(data, kernel_size=3):
    """Apply erosion followed by dilation, also called "opening", to an image or volume.
    Used to remove pixel noise by getting rid of disparate pixels that are not part of a
    larger shape.

    Args:
         data (array-like): the input 2D or 3D data to transform
         kernel_size: the size of the erosion/dilation kernel

    Returns:
        (array-like) a transformed copy of the input data that has been "opened"

    References:
        - [Opening](
           https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#opening)
    """
    return _morphological_transform(
        data=data, kernel_size=kernel_size, transform=cv2.MORPH_OPEN
    )


def dilate_and_erode(data, kernel_size=3):
    """Apply dilation followed by erosion, also called "closing", to an image or volume.
    Used to remove pixel noise by filling holes in shapes.

    Args:
         data (array-like): the input 2D or 3D data to transform
         kernel_size: the size of the erosion/dilation kernel

    Returns:
        (array-like) a transformed copy of the input data that has been "closed"

    References:
        - [Opening](
           https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#closing)
    """
    return _morphological_transform(
        data=data, kernel_size=kernel_size, transform=cv2.MORPH_CLOSE
    )


def _morphological_transform(data, kernel_size, transform):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dtype = data.dtype
    data = data.astype("float32")

    if data.ndim == 2:
        result = cv2.morphologyEx(data, transform, kernel)
    elif data.ndim == 3:
        result = data
        for dim in range(result.ndim):
            logger.info(
                f"Transforming along {result.shape[dim]} slices of dim"
                f" {dim}/{result.ndim}"
            )
            slc = [slice(None)] * result.ndim
            for idx in range(result.shape[dim]):
                slc[dim] = idx
                cur_slc = tuple(slc)
                result[cur_slc] = cv2.morphologyEx(result[cur_slc], transform, kernel)
    else:
        raise NotImplementedError(
            f"_morphological_transform() only accepts 2D or 3D data but input is"
            f" {data.ndim}D"
        )
    return result.astype(dtype)
