import os

from sacred.observers import FileStorageObserver
from sacred import Experiment

from ctseg.ctutil.utils import read_json


def initialize_experiment():
    """Initialize the Sacred Experiment

    This method reads a JSON config from mcdn-3d-seg/sacred_config.json with the
    following entries:

        experiment_name: the name of the sacred experiment
        file_observer_base_dir: the directory where run logs are saved to. If relative,
            it is assumed relative to mcdn-3d-seg/
    """
    # parse the sacred config
    repo_dir = os.path.dirname(os.path.dirname(__file__))
    sacred_config = read_json(os.path.join(repo_dir, "sacred_config.json"))

    # initialize the experiment
    ex = Experiment(sacred_config["experiment_name"])

    # create a file-based observer to log runs
    file_observer_base_dir = os.path.expanduser(sacred_config["file_observer_base_dir"])
    if not file_observer_base_dir.startswith("/"):
        file_observer_base_dir = os.path.join(repo_dir, file_observer_base_dir)
    ex.observers.append(FileStorageObserver.create(file_observer_base_dir))

    return ex


ex = initialize_experiment()


DEFAULT_CONFIG = {
    "num_gpus": 1,
    # the number of output segmentation classes
    "num_classes": 4,
    # the method used to normalize the data
    # options include: ZeroMeanUnitVar, NormLog, MagicNormLog
    "normalization": "",
    # continuously checks for new inference files and deletes completed files
    "production_mode": False,
    "check_alignment": -1,
    # model architecture
    "model_config": {
        # specifies the architecture of a new model
        "architecture_config": {
            # the size of model's input window when sampling volumes (x, y, z)
            "input_shape": [240, 240, 240],
            "kernel_initializer": "lecun_normal",
            "activation": "relu",
            "dropout_rate": 0.1,
        },
        # specifies loading a pre-trained model
        "load_config": {
            # whether or not to drop the last layer when loading a model
            "drop_last_layer": False,
            # "best", "latest" or "/PATH/TO/MODEL/CHECKPOINT" to resume training from.
            # Leave empty to not resume
            "resume_from": "",
            # path to a weights file to load the model from. takes precedent over
            # `resume_from` if set
            "load_weights_from": "",
        },
    },
    # data preprocessing
    "data_config": {
        # mirrors input chunks in the corresponding dimension
        "flip_x": False,
        "flip_y": False,
        "flip_z": False,
        # Flip Validation Axis: None or int or tuple of ints, optional
        #  Axis or axes along which to flip over. The default,
        #  axis=None, will flip over all of the axes of the input array.
        #  If axis is negative it counts from the last to the first axis.
        #  If axis is a tuple of ints, flipping is performed on all of the axes
        #  specified in the tuple.
        "flip_validation_axis": None,
        "sampler_config": {
            # the chunk sampling class for during training. one of "OverlapSampler",
            # "RandomSampler", "BattleShipSampler"
            "sampler_class": "RandomSampler",
            # Number of random samples taken from the training data dir when performing
            # training. Not used in "overlap" mode.
            "n_samples_per_epoch": 3,
            # Number of chunks taken from each sample when performing training. Not
            # used in "overlap" mode.
            "n_chunks_per_sample": 100,
            # the amount the input window is translated in the x, y, and z dimensions.
            # Used during inference but also during training if sampler_class is
            # "OverlapSampler"
            "overlap_stride": 240,
        }
    },
    # configuration specific to training
    "train_config": {
        "inputs": {
            # dir containing training the `.npy` data files
            "data_dir": "/PATH/TO/TRAIN/DATA",
            # dir containing the `.npy` training labels. files are matched by name to
            # data, so this dir can have targets for both training and testing
            "targets_dir": "/PATH/TO/TRAIN/TARGETS"
        },
        "outputs": {
            # where cached normalized data is saved to
            "normalized_data_dir": "/PATH/TO/NORMALIZED/DATA",
            "csv_log_dir":  "/PATH/TO/CSV/LOGS",
            "tensorboard_log_dir": "/PATH/TO/TENSORBOARD/LOGS",
            "models_dir": "/PATH/TO/SAVED/MODELS",
            # where normalizer metadata is saved to
            "preprocessor_dir": "/PATH/TO/SAVED/PREPROCESSORS",
        },
        "compilation": {
            # name of the optimizer to use
            "optimizer": "Adadelta",
            # the name of the loss function. Valid names include all Keras defaults
            # as well as fully-qualified function names
            "loss": "ctseg.ctutil.losses.weighted_categorical_crossentropy",
            # kwargs passed to the loss function. replace this kwargs dict with `false`
            # to not use
            "loss_kwargs": {
                "beta": 0.9,
            },
            # the names of the metrics to track. Valid names include all Keras defaults
            # as well as fully-qualified function names
            "metrics": [
                "accuracy",
                "ctseg.ctutil.metrics.per_class_accuracy",
                "ctseg.ctutil.metrics.jaccard_index",
            ],
            # indicates whether or not to recompile with the above specified optimizer,
            # loss and metrics if a compiled model is loaded.
            # Warning: doing this may slow training as it will discard the current state
            # of the optimizer
            "recompile": False
        },
        # the max number of epochs to train for
        "epochs": 1000,
        # Epoch at which to start training
        # (useful for resuming a previous training run).
        "initial_epoch": 0,
        # the training batch size
        "batch_size": 1,
    },
    # configuration specific to testing
    "test_config": {
        "inputs": {
            # dir containing the `.npy` test data files
            "data_dir": "/PATH/TO/TEST/DATA",
            # dir containing the `.npy` test labels. files are matched by name to data,
            # so this dir can have targets for both training and testing
            "targets_dir": "/PATH/TO/TEST/TARGETS"
        },
        "outputs": {
            # where cached normalized data is saved to
            "normalized_data_dir": "/PATH/TO/NORMALIZED/DATA"
        }
    },
    # configuration specific to inference
    "inference_config": {
        "inputs": {
            # where the `.npy` files to be processed live
            "unprocessed_queue_dir": "/PATH/TO/UNPROCESSED/DATA",
        },
        "outputs": {
            # where files from `unprocessed_queue_dir` are moved to once processed
            "processed_data_dir": "/PATH/TO/PROCESSED/DATA",
            # where cached normalized data is saved to
            "normalized_data_dir": "/PATH/TO/NORMALIZED/DATA",
            # where predictions are written to
            "predictions_dir": "/PATH/TO/INFERENCE/PREDICTIONS"
        },
        # the number of iterations of inference performed per chunk, the results of which
        # are averaged and standard deviations are calculated
        "inference_iters": 5,
    },
    # configuration specific to plotting
    "plot_config": {
        "inputs": {
            # dir containing the `.npy` data files
            "data_dir": "/PATH/TO/DATA",
            # dir containing the `.npy` labels, if available. Leave empty if not. Files
            # are matched by name to data
            "targets_dir": "/PATH/TO/TARGETS",
            # dir containing the `.npy` predictions
            "predictions_dir": "/PATH/TO/PREDICTIONS"
        },
        "outputs": {
            "plots_dir": "/PATH/TO/OUTPUT/PLOTS"
        }
    },
}

ex.add_config(DEFAULT_CONFIG)


@ex.named_config
def use_8_gpus():
    num_gpus=8
    batch_size=8


@ex.named_config
def use_2_gpus():
    num_gpus=2
    batch_size=2


@ex.named_config
def small_chunks():
    name="sm33_small_chunk"
    x_max=192
    y_max=192
    z_max=192
    overlap_stride=192


@ex.named_config
def small_testing():
    num_chunks_per_training_img=20
    num_training_imgs_per_epoch=1
