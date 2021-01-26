"""

"""
from collections import defaultdict
import logging
import os

from ctseg.config import DEFAULT_CONFIG


logger = logging.getLogger(__name__)


def validate_schema(d):
    """Validate the schema of a config dict against the DEFAULT_CONFIG

    Args:
        d (dict): the config to validate

    Raises:
        AssertionError if an unknown parameter is found
    """
    _validate_schema(d, default_config=DEFAULT_CONFIG, parent_key="")


def _validate_schema(d, default_config, parent_key):
    for k, v in d.items():
        full_key = ".".join((parent_key, k)) if parent_key else k

        assert k in default_config, f"Unknown parameter: '{full_key}'"

        if isinstance(v, dict):
            _validate_schema(v, default_config=default_config[k], parent_key=full_key)


def update_schema_v0_to_v1(d):
    """
    update from commit 9deac3bfb3f4e8cf1ec6876332a223c89a8c6eb6
    to commit 03ebc720f0424b00deb184038f1bc2cb258ab0fd
    """
    input_dir = d.pop("input_dir")
    data_name = d.pop("data_name")
    set_name = d.pop("set_name")
    target_name = d.pop("target_name")
    name = d.pop("name")

    # unused parameters (in the previous version of the code these parameters weren't
    # used!
    for param in [
        "training_steps_per_epoch",
        "validation_steps_per_epoch",
        "steps_per_epoch",
    ]:
        try:
            d.pop(param)
        except KeyError:
            logger.warning(f"Found unused parameter `{param}`. Discarding")

    top_dir = os.path.join(input_dir, data_name, "sets")
    # input dirs
    ct_images_train = os.path.join(top_dir, set_name, "images_train")
    ct_targets_train = os.path.join(top_dir, set_name, target_name)
    ct_images_test = os.path.join(top_dir, set_name, "images_test")
    ct_targets_test = os.path.join(top_dir, set_name, target_name)
    inference_images = os.path.join(top_dir, set_name, "infer")
    inference_complete = os.path.join(top_dir, set_name, "infer_complete")
    normalized_image_dir = os.path.join(top_dir, set_name, "normalized_images")

    try:
        top_output_dir = d.pop("top_output_dir")
        logger.warning(
            "Reading `top_output_dir` explicitly. This ignores the definition of"
            " `output_dir`"
        )
        # set `output_dir` for `csv_log_dir`
        output_dir = top_output_dir
    except KeyError:
        output_dir = d.pop("output_dir")
        top_output_dir = os.path.join(output_dir, data_name, set_name, name)

    # output dirs
    tensorboard_log_dir = os.path.join(top_output_dir, "logs")
    prediction_dir = os.path.join(top_output_dir, "predictions")
    models_dir = os.path.join(top_output_dir, "models")
    plots_dir = os.path.join(top_output_dir, "plots")

    d["train_config"] = {
        "inputs": {
            # dir containing training the `.npy` data files
            "data_dir": ct_images_train,
            # dir containing the `.npy` training labels. files are matched by name to
            # data, so this dir can have targets for both training and testing
            "targets_dir": ct_targets_train,
        },
        "outputs": {
            # where cached normalized data is saved to
            "normalized_data_dir": normalized_image_dir,
            "csv_log_dir": os.path.join(output_dir, "training.log"),
            "tensorboard_log_dir": tensorboard_log_dir,
            "models_dir": models_dir,
        },
    }

    # configuration specific to testing
    d["test_config"] = {
        "inputs": {
            # dir containing the `.npy` test data files
            "data_dir": ct_images_test,
            # dir containing the `.npy` test labels. files are matched by name to data,
            # so this dir can have targets for both training and testing
            "targets_dir": ct_targets_test,
        },
        "outputs": {
            # where cached normalized data is saved to
            "normalized_data_dir": normalized_image_dir
        },
    }

    # configuration specific to inference
    d["inference_config"] = {
        "inputs": {
            # where the `.npy` files to be processed live
            "unprocessed_queue_dir": inference_images,
        },
        "outputs": {
            # where files from `unprocessed_queue_dir` are moved to once processed
            "processed_data_dir": inference_complete,
            # where cached normalized data is saved to
            "normalized_data_dir": normalized_image_dir,
            # where predictions are written to
            "predictions_dir": prediction_dir,
        },
    }

    # configuration specific to plotting
    d["plot_config"] = {
        "inputs": {
            # dir containing the `.npy` data files
            "data_dir": inference_complete,
            # dir containing the `.npy` labels, if available. Leave empty if not. Files
            # are matched by name to data
            "targets_dir": ct_targets_test,
            # dir containing the `.npy` predictions
            "predictions_dir": prediction_dir,
        },
        "outputs": {"plots_dir": plots_dir},
    }

    return d


def update_schema_v1_to_v2(d):
    """
    update from commit 03ebc720f0424b00deb184038f1bc2cb258ab0fd
    to commit eda6df7e8a2db3136f86836a8fe0d25e60a2ef5a
    """
    d = defaultdict(dict, d)

    # Model Config
    d["model_config"] = {"architecture_config": {}, "load_config": {}}
    d2 = d["model_config"]["architecture_config"]
    input_shape_keys = ("x_max", "y_max", "z_max")
    input_shape_present = (k in d for k in input_shape_keys)
    if all(input_shape_present):
        d2["input_shape"] = [d.pop(k) for k in input_shape_keys]
    else:
        assert not any(input_shape_present), "Input shape partially specified!"
    _move_keys(d, d2, ("kernel_initializer", "activation", "dropout_rate"))
    _move_keys(
        d,
        d["model_config"]["load_config"],
        ("drop_last_layer", "resume_from", "load_weights_from"),
    )

    # Data Config
    d["data_config"] = {"sampler_config": {}}
    _move_keys(
        d, d["data_config"], ("flip_x", "flip_y", "flip_z", "flip_validation_axis")
    )
    d2 = d["data_config"]["sampler_config"]
    if "sample_mode" in d:
        sample_mode = d.pop("sample_mode")
        sampler_class_map = {
            "random": "RandomSampler",
            "overlap": "OverlapSampler",
            "bship": "BattleShipSampler",
        }
        d2["sampler_class"] = sampler_class_map[sample_mode]
    _move_keys(
        d,
        d2,
        (
            ("num_training_imgs_per_epoch", "n_samples_per_epoch"),
            ("num_chunks_per_training_img", "n_chunks_per_sample"),
            "overlap_stride",
        ),
    )

    # Train Config
    _move_keys(d, d["train_config"], ("epochs", "initial_epoch", "batch_size"))

    # Inference Config
    _move_keys(d, d["inference_config"], ("inference_iters",))

    return d


def _move_keys(d1, d2, keys):
    """Move move keys from one dict to another. `keys` should be an iterable of key
    strings and/or (key1, key2) tuples
    """
    for k in keys:
        # split the key if it is a tuple
        k1, k2 = (k, k) if isinstance(k, str) else k
        if k1 in d1:
            d2[k2] = d1.pop(k1)


SCHEMA_UPDATES = [update_schema_v0_to_v1, update_schema_v1_to_v2]
