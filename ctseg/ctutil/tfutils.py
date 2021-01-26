"""tfutils.py
Tensorflow Utility Functions

> Tyler Ganter, tganter@sandia.gov
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import keras.backend as K
    except ModuleNotFoundError:
        from tensorflow.keras import backend as K

    import tensorflow as tf


def ndim(x):
    """Returns the ndim of a TF tensor or numpy array"""
    try:
        return K.ndim(x)
    except AttributeError:
        pass
    return x.ndim


def label_freq(y_true, keepdims=False):
    """Label Frequency

    Args:
        y_true: Tensor of one-hot true targets.
            shape = `[batch_size, d0, .. dN, num_classes]`.
        keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced to 2, i.e.
          (batch_size, num_classes) otherwise, the reduced dimensions are retained with
          length 1.

    Returns:
        a Tensor of shape: (batch_size, ...<KEPT_DIMS>, num_classes)
    """
    # do not sum over 1st (batch) or last (label) axes
    sum_axes = tuple(range(1, ndim(y_true) - 1))

    # count the number of pixels per class
    label_counts = K.sum(y_true, axis=sum_axes, keepdims=keepdims)

    # frequency per class
    return label_counts / K.sum(label_counts, axis=-1, keepdims=True)


def inv_label_freq(y_true, beta=0.9, rescale=True, keepdims=False):
    """Inverse Label Frequency

    Args:
        y_true: Tensor of one-hot true targets.
            shape = `[batch_size, d0, .. dN, num_classes]`.
        beta (float): compression hyperparameter in range [0, 1]. Assuming the inverse
            label frequency is used for loss weighting,
                beta = 0 corresponds to no re-weighting
                beta = 1 corresponds to re-weighing by inverse class frequency
        rescale (bool): whether or not to rescale the inverse label frequency such
            that the sum across each batch is equal to num_classes. Assuming this is
            used for loss weighting, this will keep the total loss roughly the same
            scale
        keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced to 2, i.e.
          (batch_size, num_classes) otherwise, the reduced dimensions are retained with
          length 1.

    Returns:
        a Tensor of shape: (batch_size, ...<KEPT_DIMS>, num_classes)

    References:
        - [Class-Balanced Loss Based on Effective Number of Samples](
           https://arxiv.org/pdf/1901.05555.pdf)
    """
    freqs = label_freq(y_true, keepdims=keepdims) + K.epsilon()

    if beta == 1:
        inv_freqs = 1 / freqs
    else:
        # compress weights
        inv_freqs = (1 - beta) / (1 - beta ** freqs + K.epsilon())

    if rescale:
        num_classes = tf.cast(K.shape(inv_freqs)[-1], inv_freqs.dtype)
        inv_freqs *= num_classes / K.sum(inv_freqs, axis=-1, keepdims=True)

    return inv_freqs
