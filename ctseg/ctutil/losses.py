"""losses.py
Loss Functions

> Tyler Ganter, tganter@sandia.gov
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import keras
        import keras.backend as K
        from keras.backend import log
    except ModuleNotFoundError:
        from tensorflow import keras
        from tensorflow.keras import backend as K
        from tensorflow.math import log

    import tensorflow as tf

from ctseg.ctutil.tfutils import inv_label_freq


def weighted_binary_crossentropy(y_true, y_pred, beta=0.9):
    """Binary CrossEntropy weighted by inverse label frequency.

    Args:
        y_true: Ground truth values. shape = `(batch_size, d0, .. dN)`.
        y_pred: The predicted values. shape = `(batch_size, d0, .. dN)`.
        beta (float): compression hyperparameter in range [0, 1].
             beta = 0 corresponds to no re-weighting
             beta = 1 corresponds to re-weighing by inverse class frequency

    Returns:
        Weighted Binary crossentropy loss value. shape = `(batch_size, d0, .. dN-1)`.
    """
    if tf.__version__[0] != "1":
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

    # standard binary crossentropy
    # Note: using `keras.backend` here, not `keras.losses`, because the `keras.losses`
    # version applies a mean over the last axis, which must be applied AFTER weighting
    bce = K.binary_crossentropy(y_true, y_pred)

    # inverse label frequency weighting
    y_true_one_hot = K.one_hot(tf.cast(y_true, tf.uint8), 2)
    weights = inv_label_freq(y_true_one_hot, beta=beta, keepdims=True)
    masked_w = tf.cast(K.sum(y_true_one_hot * weights, axis=-1), y_pred.dtype)

    return K.mean(bce * masked_w, axis=-1)


def weighted_categorical_crossentropy(y_true, y_pred, beta=0.9):
    """Computes the categorical crossentropy loss.

    Args:
        y_true: Tensor of one-hot true targets.
            shape = `(batch_size, d0, .. dN, num_classes)`.
        y_pred: Tensor of predicted targets.
            shape = `(batch_size, d0, .. dN, num_classes)`.
        beta (float): compression hyperparameter in range [0, 1].
             beta = 0 corresponds to no re-weighting
             beta = 1 corresponds to re-weighing by inverse class frequency

    Returns:
        Weighted Categorical crossentropy loss value.
            shape = `(batch_size, d0, .. dN)`.
    """
    if tf.__version__[0] != "1":
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

    # standard categorical crossentropy
    cce = keras.losses.categorical_crossentropy(y_true, y_pred)

    # inverse label frequency weighting
    weights = inv_label_freq(y_true, beta=beta, keepdims=True)

    # mask weights by ground truth
    masked_w = K.sum(y_true * weights, axis=-1)

    return cce * masked_w


# Used when loading a compiled model via keras.models.load_model()
LOSS_MAP = {
    weighted_binary_crossentropy.__name__: weighted_binary_crossentropy,
    weighted_categorical_crossentropy.__name__: weighted_categorical_crossentropy,
}
