# Luxonis training library

Luxonis training framework (`luxonis-train`) is intended for training deep learning models that can run fast on OAK products.

The project is in alpha state - please report any feedback.

**Table of contents:**

- [Luxonis training library](#luxonis-training-library)
  - [Installation](#installation)
  - [Training](#training)
  - [Customizations](#customizations)
  - [Tuning](#tuning)
  - [Exporting](#exporting)
  - [Credentials](#credentials)

## Installation

`luxonis-train` is hosted on PyPi and can be installed with `pip` as:

```bash
pip install luxonis-train[all]
```

## Usage

The entire configuration is specified in a `yaml` file. This includes the model
structure, used losses, metrics, optimizers etc. For specific instructions and example
configuration files, see [Configuration](./configs/README.md).



## Training

Once you've configured your `config.yaml` file you can train the model using this command:

```bash
python3 -m luxonis_train train --config config.yaml
```

If you wish to manually override some config parameters you can do this by providing the key-value pairs. Example of this is:

```bash
python3 -m luxonis_train train --config config.yaml train.batch_size 8 train.epochs 10
```

where key and value are space separated and sub-keys are dot (`.`) separated. If the configuration field is a list, then key/sub-key should be a number (e.g. `train.preprocessing.augmentations.0.name RotateCustom`).

## Customizations

We provide a module registry interface through which you can create new backbones, necks, heads, losses, callbacks, optimizers and scheduler. This can be then referenced in the config and used during training. To ensure that everything still works correctly we recommend you inherit from specific base classes:

- Node - [LuxonisNode](src/luxonis_train/models/nodes/luxonis_node.py)
- Loss - [LuxonisLoss](src/luxonis_train/attached_modules/losses/luxonis_loss.py)
- Metric - [LuxonisMetric](src/luxonis_train/attached_modules/metrics/luxonis_metric.py)
- Visualizer - [LuxonisVisualizer](src/luxonis_train/attached_modules/visualizers/luxonis_visualizer.py)
- Callback - [Callback from lightning.pytorch.callbacks](lightning.pytorch.callbacks)
  - This are referenced through Config in `train.callbacks.custom_callbacks` similarly as augmentations (list of dictionaries with module name and params)
- Optimizer - [Optimizer from torch.optim](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
- Scheduler - [LRScheduler from torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

Here is an example of how to create a custom components:

```python
from torch.optim import Optimizer
from luxonis_train.utils.registry import OPTIMIZERS
from luxonis_train.attached_modules.losses import LuxonisLoss

@OPTIMIZERS.register_module()
class CustomOptimizer(Optimizer):
    ...

# Subclasses of LuxonisNode, LuxonisLoss, LuxonisMetric
# and LuxonisVisualizer are registered automatically.

class CustomLoss(LuxonisLoss):
    # This class is automatically registered under `CustomLoss` name.
    def __init__(self, k_steps: int):
        ...
```

And then in the config you reference this `CustomOptimizer` and `CustomLoss` by their names.

```yaml
losses:
  - name: CustomLoss
    params:  # additional parameters
      k_steps: 12

```

## Tuning

To improve training performance you can use `Tuner` for hyperparameter optimization. There are some study related parameters that you can change for initialization. To do the actual tuning you have to setup a `tuner.params` block in the config file where you specify which parameters to tune and the ranges from which the values should be chosen. An example of this block is shown below. The key should be in the format: `key1.key2.key3_<type>` where type can be one of `[categorical, float, int, loguniform, uniform]` (for more information about specific type check out [Optuna documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)).

```yaml
tuner:
  study_name: "test-study" # name of the study (string)
  use_pruner: True # if should use MedianPruner (bool)
  n_trials: 3 # number of trials for each process (int)
  timeout: 600 # stop study after the given number of seconds (int)
  storage:
    active: True # if should use storage to make study persistent (bool)
    type: local # type of storage, "local" or "remote" (string)
  params: # (key, value) pairs for tuning
```

Example of params for tuner block:

```yaml
tuner:
  params:
    train.optimizers.optimizer.name_categorical: ["Adam", "SGD"]
    train.optimizers.optimizer.params.lr_float: [0.0001, 0.001]
    train.batch_size_int: [4, 4, 16]
```

When a `tuner` block is specified, you can start tuning like:

```bash
python3 -m luxonis_train tune --config config.yaml
```

## Exporting

We support export to `ONNX`, `OpenVINO`, and `DepthAI .blob format` which is used for OAK cameras. By default, we only export to ONNX, but you can also export to the other two formats by changing the config (see below). We are developing a separate tool for model converting which will support even more formats (TBA).

For export you must use the same `model` configuration as in training in addition to `exporter` block in config. In this block you must define `export_weights`, other parameters are optional and can be left as default.

There is also an option to upload .ckpt, .onnx and config.yaml files to remote storage. If this is active you need to specify `upload_directory`. See [here](#paths) for examples but note that local path is not valid in this case. **Note**: This path can be overridden by `train.callbacks.export_on_finish` callback.

```yaml
exporter:
  export_weights: null # path to local weights used for export (string)
  export_save_directory: output_export # path to save directory of exported models (string)
  export_image_size: [256, 256] # Image size used for export [height, width] (list). Will be inferred from the validation dataloader if not specified.
  export_model_name: model # name of the exported model (string)
  data_type: FP16 # data type used for openvino conversion (string)
  reverse_input_channels: True # bool if reverse input shapes (bool)
  scale_values: [58.395, 57.120, 57.375] # list of scale values (list[int|float]). Inferred from augmentations if not specified.
  mean_values: [123.675, 116.28, 103.53] # list of mean values (list[int|float]). Inferred from augmentations if not specified.
  onnx:
    opset_version: 12 # opset version of onnx used (int)
    dynamic_axes: null # define if dynamic input shapes are used (dict)
  blobconverter:
    active: False # bool if export to blob (bool)
    shaves: 6 # number of shaves used (int)
  upload: # uploads .ckpt, .onnx, config file and modelconverter config file
    active: False  # bool if upload active (bool)
    upload_directory: null # either path to s3 or existing mlflow run (string)
```

Once you have the config file ready you can export the model like this:

```bash
python3 -m luxonis_train export --config config.yaml
```

## Credentials

By default local use is supported. But we also integrate some cloud services which can be primarily used for logging and storing. When these are used, you need to load environment variables to set up the correct credentials.

If you are working with LuxonisDataset that is hosted on S3 you need to specify these env variables:

```bash
AWS_ACCESS_KEY_ID=**********
AWS_SECRET_ACCESS_KEY=**********
AWS_S3_ENDPOINT_URL=**********
```

If you want to use MLFlow for logging and storing artifacts you also need to specify MLFlow-related env variables like this:

```bash
MLFLOW_S3_BUCKET=**********
MLFLOW_S3_ENDPOINT_URL=**********
MLFLOW_TRACKING_URI=**********
```

And if you are using WanDB for logging you have to sign in first in your environment.

Lastly, there is an option for remote storage when using `Tuner`. Here we use POSTGRES and to connect to the database you need to specify the folowing env variables:

```bash
POSTGRES_USER=**********
POSTGRES_PASSWORD=**********
POSTGRES_HOST=**********
POSTGRES_PORT=**********
POSTGRES_DB=**********
```
