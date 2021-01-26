"""metrics.py
Model Evaluation Metrics

> Tyler Ganter, tganter@sandia.gov
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import keras.backend as K
    except ModuleNotFoundError:
        from tensorflow.keras import backend as K

from ctseg.ctutil.losses import LOSS_MAP
from ctseg.ctutil.tfutils import ndim


def per_class_accuracy(y_true, y_pred):
    """Computes the average per-class accuracy

    Args:
        y_true: The ground truth tensor. shape = `(batch_size, d0, .. dN, num_classes)`.
        y_pred: The predicted tensor. shape = `(batch_size, d0, .. dN, num_classes)`.

    Returns:
        a tensor with shape = `(batch_size,)`

    References:
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    # do not sum over 1st (batch) or last (label) axes
    sum_axes = tuple(range(1, ndim(y_true) - 1))

    # count correctly labeled pixels per class
    intersection = K.sum(y_true * y_pred, axis=sum_axes)

    # count pixels per class
    label_counts = K.sum(y_true, axis=sum_axes)

    # batch_size X num_classes tensor
    per_class_accs = (intersection + K.epsilon()) / (label_counts + K.epsilon())

    return K.mean(per_class_accs, axis=-1)


# @todo(Tyler) ...could this be replaced with keras.metrics.MeanIoU?
def jaccard_index(y_true, y_pred):
    """Computes the Jaccard Index for semantic segmentation, also known as the
    intersection-over-union.

    Args:
        y_true: The ground truth tensor. shape = `(batch_size, d0, .. dN, num_classes)`.
        y_pred: The predicted tensor. shape = `(batch_size, d0, .. dN, num_classes)`.

    Returns:
        a tensor with shape = `(batch_size,)`

    References:
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    # do not sum over 1st (batch) or last (label) axes
    sum_axes = tuple(range(1, ndim(y_true) - 1))

    # count correctly labeled pixels per class
    intersection = K.sum(y_true * y_pred, axis=sum_axes)

    # count pixels per class
    label_counts = K.sum(y_true, axis=sum_axes)

    # count pixels per class
    pred_counts = K.sum(y_pred, axis=sum_axes)

    jac = (intersection + K.epsilon()) / (
        label_counts + pred_counts - intersection + K.epsilon()
    )

    return K.mean(jac, axis=-1)


METRIC_MAP = {
    per_class_accuracy.__name__: per_class_accuracy,
    jaccard_index.__name__: jaccard_index,
}


# When loading a compiled model, pass this map to:
#   keras.models.load_model(..., custom_objects=LOSS_METRIC_MAP)
LOSS_METRIC_MAP = {**LOSS_MAP, **METRIC_MAP}
if len(LOSS_METRIC_MAP) != (len(LOSS_MAP) + len(METRIC_MAP)):
    raise NameError("Name collision between losses and metrics")
