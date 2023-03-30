"""Core Utility Functions"""
import json
import logging
import os
import sys

import numpy as np


logger = logging.getLogger(__name__)


def read_json(path):
    """Reads JSON from file.

    Args:
        path: the path to the JSON file

    Returns:
        a dict or list containing the loaded JSON

    Raises:
        ValueError: if the JSON file was invalid
    """
    try:
        with open(path, "rt") as f:
            return json.load(f)
    except ValueError:
        raise ValueError("Unable to parse JSON file '%s'" % path)


def write_json(obj, path, pretty_print=False):
    """Writes JSON object to file, creating the output directory if necessary.

    Args:
        obj: an object that can be directly dumped to a JSON file
        path: the output path
        pretty_print: whether to render the JSON in human readable format with
            newlines and indentations. By default, this is False
    """
    # create dirs if necessary
    dirname = os.path.dirname(path)
    if dirname and not os.path.isdir(dirname):
        os.makedirs(dirname)

    # write the output file
    s = json_to_str(obj, pretty_print=pretty_print)
    with open(path, "wt") as f:
        f.write(s)


def json_to_str(obj, pretty_print=True):
    """Converts the JSON object to a string.

    Args:
        obj: a JSON dictionary
        pretty_print: whether to render the JSON in human readable format with
            newlines and indentations. By default, this is True
    """
    kwargs = {"indent": 4} if pretty_print else {}
    s = json.dumps(obj, separators=(",", ": "), ensure_ascii=False, **kwargs)
    return str(s)


def describe_array(x, downsample=-1):
    """Describe a numpy array

    Args:
        x (array-like): the array to describe
        downsample (int): step to downsample when computing stats (for efficiency)
            If downsample == -1 and the array is "large", it will be automatically
            downsampled. To disable this, set downsample=1.

    Returns:
        a dictionary of metadata and statistics
    """
    MAX_SIZE = 25e7

    d = {
        "dtype": x.dtype,
        "shape": x.shape,
    }

    if downsample == -1:
        # automatically select downsample step if data is too large
        size = np.prod(x.shape)

        if size > MAX_SIZE:
            # downsample to ~MAX_SIZE
            downsample = int(np.round((size / MAX_SIZE) ** 1 / 3))

    # downsample data before computing stats
    if downsample > 1:
        slc = (slice(None, None, downsample),) * len(x.shape)
        x = x[slc]

    d["stats_downsample_rate"] = downsample
    d["mean"] = np.mean(x)
    d["std"] = np.std(x)
    d["min"] = np.min(x)
    d["max"] = np.max(x)

    return d


def onehot_encode(x, num_classes=None):
    """Onehot encodes an array of integer class labels

    Args:
        x (array-like): an array of integer class labels (or castable to integer)
        num_classes (int): the number of classes to encode. Inferred from the data if
        not provided.

    Returns:
        (array-like): an array of 0s and 1s of shape (...<x dims>, num_classes), i.e.
            one extra dimension equal to the number of classes
    """
    x = x.astype(np.uint8)
    if num_classes is None:
        num_classes = np.max(x) + 1
    return np.eye(num_classes, dtype=np.uint8)[x]


def get_class(class_name, module_name=None):
    """Returns the class specified by the given class string, loading the
    parent module if necessary.

    Args:
        class_name: the "ClassName" or a fully-qualified class name like
            "eta.core.utils.ClassName"
        module_name: the fully-qualified module name like "ctseg.ctutil.utils", or
            None if class_name includes the module name. Set module_name to
            __name__ to load a class from the calling module

    Returns:
        the class

    Raises:
        ImportError: if the class could not be imported
    """
    if module_name is None:
        try:
            module_name, class_name = class_name.rsplit(".", 1)
        except ValueError:
            raise ImportError(
                "Class name '%s' must be fully-qualified when no module "
                "name is provided" % class_name
            )

    __import__(module_name)  # does nothing if module is already imported
    return getattr(sys.modules[module_name], class_name)


def get_function(function_name, module_name=None):
    """Returns the function specified by the given string.
    Loads the parent module if necessary.

    Args:
        function_name: local function name by string fully-qualified name
            like "eta.core.utils.get_function"
        module_name: the fully-qualified module name like "ctseg.ctutil.utils", or
            None if function_name includes the module name. Set module_name to
            __name__ to load a function from the calling module

    Returns:
        the function

    Raises:
        ImportError: if the function could not be imported
    """
    return get_class(function_name, module_name=module_name)
