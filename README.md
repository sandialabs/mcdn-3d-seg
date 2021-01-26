# Monte Carlo Dropout Network (MCDN) 3D CT Segmentation

A Tensorflow and Keras backed framework for learned segmentation methods of 3D CT scan
volumes. Supported functionality includes training models, running inference and
quantifying uncertainty. The main underlying model architecture is V-Net.

## Installation

`mcdn-3d-seg` expects Python `>=3.5`. Also, be sure to activate your desired
virtual environment before installing.

Run one of the following to install, depending on if you want GPU support:

```bash
# for GPU support
pip install -e .[gpu]

# for CPU only
pip install -e .[cpu]
```

## Sacred Configuration

The JSON file `sacred_config.json` specifies [sacred](sacred.readthedocs.io) experiment
configuration independent of the run configuration.

Specifically, `file_observer_base_dir` specifies where sacred stores its run logs. The
default is `mcdn-3d-seg/runs/` but if running on synapse, the shared logs are
stored to:

```json
{
    "file_observer_base_dir": "/data/wg-cee-dev-dgx/output/ct_seg/file_observer"
}
```

## Run Configuration

`mcdn-3d-seg` contains multiple scripts such as `train.py` and `infer.py`, each
of which expects a JSON config. The syntax to run one of these scripts is:

```bash
python <PATH/TO/SCRIPT>.py with <PATH/TO/CONFIG>.json
```

### Creating a JSON config

The default config is a python `dict`, `DEFAULT_CONFIG`, in the file `ctseg/config.py`.
The JSON config needs to match the syntax of `DEFAULT_CONFIG` but only needs to contain
override values.

1. create an empty JSON file
2. add individual fields from `DEFAULT_CONFIG` or copy/paste the entire config into your
JSON file
3. modify values as appropriate

**Important**: be sure to update the inputs and outputs as the default values are just
placeholders.

## Training

Important config parameters:

- `normalization`: the default is no normalization but you will likely want to change
this
- `train_config`, `test_config`: the inputs and outputs here need to be specified for
every experiment.

**Note**: `inference_config` and `plot_config` are not used during training.

Once a config has been created, use the config to train via:

```bash
python train.py with <PATH/TO/CONFIG>.json
```

### Resuming Training

1. Select whether you want to resume from the `best` model or the `latest` model. If you saved
the entire model (default), then use `resume_from`; otherwise use `load_weights_from`  
2. Resume training the model:

```bash
python train.py with <PATH/TO/CONFIG>.json resume_from=<"best" OR "latest">`
```
   
## Inference

Before running inference, be sure you have specified the correct input/output paths in
`inference_config` of the JSON config. Once set, run inference via:    

```bash
python infer.py with <PATH/TO/CONFIG>.json
```

To run inference using a different model, add `resume_from=<PATH/TO/MODEL>` at the end of the
above command.
