import logging
import os

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, Input, add, concatenate, SpatialDropout3D
from keras.utils import multi_gpu_model
import numpy as np
from tqdm import tqdm

from ctseg.ctutil.utils import get_function

from ctseg.ctutil.metrics import LOSS_METRIC_MAP


logger = logging.getLogger(__name__)


def create_vnet(
    input_shape,
    num_classes,
    kernel_initializer="lecun_normal",
    activation="relu",
    dropout_rate=0.1,
):
    """Initializes a VNet model

    Args:
        input_shape: tuple or list of (x, y, z) input dimensions
        num_classes: the number of classes, i.e. length of one-hot encoded vectors
        kernel_initializer: Regularizer function applied to the `kernel` weights
            matrix
        activation: Activation function to use
        dropout_rate: float between 0 and 1. Fraction of the input units to drop.
    """
    logger.info("initializing a new VNet model")
    K.clear_session()

    # Layer 1
    input_layer = Input(shape=(*input_shape, 1), name="data")
    main = Conv3D(
        8,
        (5, 5, 5),
        padding="same",
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(input_layer)
    main = left1 = add([main, input_layer])
    main = Conv3D(
        16,
        (2, 2, 2),
        strides=(2, 2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(main)

    # Layer 2,3,4
    main, left2 = downward_layer(
        main, 2, 32, kernel_initializer=kernel_initializer, activation=activation
    )
    main, left3 = downward_layer(
        main, 2, 64, kernel_initializer=kernel_initializer, activation=activation
    )
    main, left4 = downward_layer(
        main, 2, 128, kernel_initializer=kernel_initializer, activation=activation
    )
    out4 = main

    # Layer 5
    for _ in range(5):
        main = Conv3D(
            128,
            (3, 3, 3),
            padding="same",
            kernel_initializer=kernel_initializer,
            activation=activation,
        )(main)
    main = add([main, out4])
    main = keras.layers.Conv3DTranspose(
        64,
        (2, 2, 2),
        strides=(2, 2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(main)

    # Layer 6,7,8
    main = upward_layer(
        main,
        left4,
        3,
        32,
        kernel_initializer=kernel_initializer,
        activation=activation,
        dropout_rate=dropout_rate,
    )
    main = upward_layer(
        main,
        left3,
        3,
        16,
        kernel_initializer=kernel_initializer,
        activation=activation,
        dropout_rate=dropout_rate,
    )
    main = upward_layer(
        main,
        left2,
        2,
        8,
        kernel_initializer=kernel_initializer,
        activation=activation,
        dropout_rate=dropout_rate,
    )

    # Layer 9
    merged = concatenate([main, left1])
    main = Conv3D(
        16,
        (5, 5, 5),
        padding="same",
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(merged)
    main = add([main, merged])
    out = Conv3D(num_classes, (1, 1, 1), padding="same", activation="softmax")(main)

    return Model(input_layer, out)


def downward_layer(inp, n_conv, channels, kernel_initializer, activation):
    out = inp
    for _ in range(n_conv):
        out = Conv3D(
            channels // 2,
            (5, 5, 5),
            padding="same",
            kernel_initializer=kernel_initializer,
            activation=activation,
        )(out)
        # Web search suggests that dropping out on downsample is dangerous as
        # it leaves reduced data for later layers
        # if dropout_rate:
        #    out = SpatialDropout3D(dropout_rate)(out, training=True)
    out = add([out, inp])
    downsampled = Conv3D(
        channels,
        (2, 2, 2),
        padding="valid",
        kernel_initializer=kernel_initializer,
        activation=activation,
        strides=(2, 2, 2),
    )(out)
    # if dropout_rate:
    #    downsampled = SpatialDropout3D(dropout_rate)(downsampled, training=True)
    return downsampled, out


def upward_layer(
    skip_connection,
    inp,
    n_conv,
    channels,
    kernel_initializer,
    activation,
    dropout_rate,
):
    out = merged = concatenate([skip_connection, inp])
    for _ in range(n_conv):
        out = Conv3D(
            channels * 4,
            (5, 5, 5),
            padding="same",
            kernel_initializer=kernel_initializer,
            activation=activation,
        )(out)
        if dropout_rate:
            out = SpatialDropout3D(dropout_rate)(out, training=True)
    out = add([merged, out])
    out = keras.layers.Conv3DTranspose(
        channels,
        (2, 2, 2),
        strides=(2, 2, 2),
        kernel_initializer=kernel_initializer,
        activation=activation,
    )(out)
    if dropout_rate:
        out = SpatialDropout3D(dropout_rate)(out, training=True)
    return out


def path_to_model_file(filename, models_dir, normalization):
    return os.path.join(models_dir, normalization + filename)


def extract_from_multigpu_model(model):
    try:
        model = model.get_layer("model_1")
    except ValueError:
        pass
    try:
        model = model.get_layer("model_2")
    except ValueError:
        pass
    print("extracted from multigpu model")
    return model


def load_model(model_config, train_config, num_classes, normalization, compile=True):
    """

    Args:
        model_config:
        train_config:
        num_classes:
        normalization:
        compile (bool): Whether or not to compile the model after loading. This should
            be set to True if the model is going to be fine-tuned and it is desired to
            retain the training state. It should be False if only using for inference.

    Returns:
        the loaded model object
    """
    logger.info("loading a saved model")
    models_dir = train_config["outputs"]["models_dir"]

    architecture_config = model_config["architecture_config"]
    drop_last_layer = model_config["load_config"]["drop_last_layer"]
    load_weights_from = model_config["load_config"]["load_weights_from"]
    # default to "best" if not set
    resume_from = model_config["load_config"]["resume_from"] or "best"

    if load_weights_from:
        model_w = _load_model(
            load_weights_from, compile=compile, custom_objects=LOSS_METRIC_MAP
        )
        model_w = extract_from_multigpu_model(model_w)
        if drop_last_layer:
            out = Conv3D(
                num_classes,
                (1, 1, 1),
                padding="same",
                activation="softmax",
                name="last_layer",
            )(model_w.layers[-2].output)
            model_w = Model(inputs=[model_w.input], outputs=[out])
        fname = load_weights_from + ".weights"
        model_w.save_weights(fname)

        model = create_vnet(**architecture_config, num_classes=num_classes)
        model.load_weights(fname, by_name=True)
    else:
        if resume_from in ("best", "latest"):
            filename = path_to_model_file(
                resume_from + ".hdf5", models_dir, normalization
            )
        else:
            filename = resume_from
        print("Loading from", filename)
        model = _load_model(filename, compile=compile, custom_objects=LOSS_METRIC_MAP)
        model = extract_from_multigpu_model(model)

    if drop_last_layer:
        out = Conv3D(
            num_classes,
            (1, 1, 1),
            padding="same",
            activation="softmax",
            name="last_layer",
        )(model.layers[-2].output)
        model = Model(inputs=[model.input], outputs=[out])

    return model


def get_optimizer_string(model):
    return str(model.optimizer)[18:][:-26]


def get_loss_string(model):
    return str(model.loss)[10:][:-19]


def get_model(model_config, num_classes, train_config, normalization, num_gpus):
    architecture_config = model_config["architecture_config"]
    load_weights_from = model_config["load_config"]["load_weights_from"]
    resume_from = model_config["load_config"]["resume_from"]

    if resume_from or load_weights_from:
        model = load_model(model_config, train_config, num_classes, normalization)
    else:
        model = create_vnet(**architecture_config, num_classes=num_classes)
    if num_gpus > 1:
        model = multi_gpu_model(model, gpus=num_gpus)

    return model


def compile_model(model, recompile, optimizer, loss, loss_kwargs, metrics):
    """Compile a model

    Args:
        model: the model to compile
        recompile: indicates whether or not to recompile with the above specified
            optimizer, loss and metrics if a compiled model is loaded.
            Warning: doing this may slow training as it will discard the current state
            of the optimizer
        optimizer: name of the optimizer to use
        loss: the name of the loss function. Valid names include all Keras defaults
            as well as fully-qualified function names
        loss_kwargs: optional kwargs dict passed to loss function
        metrics: the names of the metrics to track. Valid names include all Keras
            defaults as well as fully-qualified function names
    """
    if model.optimizer and not recompile:
        # don't compile!
        return

    try:
        _loss_fn = get_function(loss)
    except ImportError:
        # could not find loss function, assuming it is a Keras native loss
        loss = "keras.losses." + loss
        _loss_fn = get_function(loss)

    if loss_kwargs:
        loss_fn = lambda y_true, y_pred: _loss_fn(y_true, y_pred, **loss_kwargs)
    else:
        loss_fn = _loss_fn

    for idx, metric in enumerate(metrics):
        try:
            metrics[idx] = get_function(metric)
        except ImportError:
            # could not find metric, assuming it is a Keras native metric
            pass

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)


def predict_stochastic_chunk(model, data, iters):
    """Predict multiple times on a single chunk of data and return the mean/standard
    deviation.

    Args:
        model: the model to predict with
        data (array-like): the input data that matches the model input shape
        iters: the number of iters to average over

    Returns:
        raw (array-like): the mean prediction
            shape: (...data.shape, num_classes)
        std (array-like): standard deviation of the prediction
            shape: (...data.shape, num_classes)
    """
    output_shape = list(model.output_shape)
    # replace batch size
    output_shape[0] = iters

    arr = np.zeros(output_shape, dtype=np.float64)

    for j in range(iters):
        arr[j] = model.predict_on_batch(data)

    raw = np.mean(arr, axis=0).astype(np.float32)
    std = np.std(arr, axis=0).astype(np.float32)

    return raw, std


def predict_stochastic(loader, model, iters):
    """Predict multiple times on data and return the mean/standard deviation. Processes
    the data in chunks and combines those chunks together.

    Args:
        loader: a :class:`DataLoader` instance
        model: the model to predict with
        iters: the number of iters to average over

    Returns:
        raw (array-like): the mean prediction
            shape: (...data.shape, num_classes)
        std (array-like): standard deviation of the prediction
            shape: (...data.shape, num_classes)
    """
    # initialize volumes
    x_size, y_size, z_size = loader.get_image(0).shape
    num_classes = model.output_shape[-1]
    raw = np.zeros((x_size, y_size, z_size, num_classes), dtype=np.float32)
    std = np.zeros((x_size, y_size, z_size, num_classes), dtype=np.float32)
    norm = np.zeros((x_size, y_size, z_size, num_classes), dtype=np.int)

    for i in tqdm(range(len(loader))):
        # @todo(Tyler) use a batch instead of a single chunk at a time
        data = loader.get_chunk(i)

        cur_raw, cur_std = predict_stochastic_chunk(model, data, iters=iters)

        x_coords, y_coords, z_coords = loader.get_coords(i)
        slc = (slice(*x_coords), slice(*y_coords), slice(*z_coords), slice(None))

        norm[slc] += 1
        raw[slc] += cur_raw.squeeze()
        std[slc] += cur_std.squeeze()

    raw /= norm
    std /= norm

    return raw, std


def _load_model(*args, **kwargs):
    """Loads model and loads without compile if initial load fails."""
    try:
        return keras.models.load_model(*args, **kwargs)
    except Exception as e:
        logger.warning(
            f"Failed to load and compile model with error: '{e}'. Attempting to load"
            f" with `compile=False`"
        )
    kwargs["compile"] = False
    return keras.models.load_model(*args, **kwargs)
