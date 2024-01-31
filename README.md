# OUTDATED

# Luxonis training library

Luxonis training library (luxonis-train) is intended for training deep learning models that can run fast on OAK products. We currently support image classification (multi-class and multi-label), semantic segmentation, object detection and keypoint detection tasks. The library also depends on `Luxonis Dataset Format`, which you can see [here](https://github.com/luxonis/luxonis-ml). The main idea is that the user can quickly build a model, that can run on edge devices, by defining model components through config or choose from predefined models, and train it.

The work on this project is in an MVP state, so it may be missing some critical features or have some issues - please report any feedback!

**Table of contents:**

- [OUTDATED!](#outdated)
- [Luxonis training library](#luxonis-training-library)
  - [Installation:](#installation)
  - [Configuration:](#configuration)
    - [Model:](#model)
      - [Paths](#paths)
      - [Advanced configuration:](#advanced-configuration)
    - [Trainer:](#trainer)
    - [Logger:](#logger)
    - [Dataset](#dataset)
    - [Train](#train)
    - [Initialization](#initialization)
  - [Training](#training)
  - [Customizations](#customizations)
  - [Tuning](#tuning)
  - [Inference](#inference)
  - [Exporting](#exporting)
  - [Test Dataset](#test-dataset)
  - [Credentials:](#credentials)

## Installation

To install this library you can commands below:

```bash
git clone -b dev git@github.com:luxonis/models.git && cd models
python3 -m pip install .
```

***Note**: This will also install `luxonis-ml` library.*

## Configuration

Most of the work is done through a `config.yaml` file, which you must pass as an argument to the Trainer. Config file consists of a few major blocks that are described below. You can create your own config or use/edit one of the already made ones.

### Model

This is the most important block, that **must be always defined by the user**. There are two different ways you can create the model.

In the first one you defined `backbone`, `neck` and list of `heads`, where you choose each module from a list of supported ones (list of all [backbones](./src/luxonis_train/models/backbones/README.md), [necks](./src/luxonis_train/models/necks/README.md) and [heads](./src/luxonis_train/models/heads/README.md)). If `n_classes` is not set for the head or is set to `null` then this value will be inherited from the dataset.

***Note**: Every model must have 1 bacbone and at least 1 head (neck is optional).*

```yaml
model:
  name: test_model
  type: # leave this empty (null)
  pretrained: # path to weights for whole model (overrides backbone weights) (string)

  backbone:
    name: # neme of the backbone (e.g. ResNet18)
    pretrained: # path to weights for backbone (string)
    params: # params specific to this backbone (dict)

  # neck is optional
  neck:
    name: # name of the neck (e.g. RepPANNeck) (string)
    params: # params specific to this neck (dict)

  # you can define multiple heads
  heads: # first head is most important, its metric used for tracking whole model performance
    - name: # name of the head (e.g. ClasificationHead) (string)
      params: # params specific to this head (dict)

      # Every head must have its loss function
      loss:
        name: # name of the loss function (e.g. CrossEntropyLoss) (string)
        params: # params specific to this loss function (dict)
```

The second use case is choosing from a list of predefined model arhitectures ([list](./src/luxonis_train/models/README.md)) and optionally adding additional heads. In this case you define model block in config like this:

```yaml
model:
  name: predefined_model
  type: # name of predefined arhitecture (e.g. YoloV6-n) (string)
  pretrained: # path to weights for whole model (string)
  params: # model-wise params (dict)

  # additional heads are optional, you can define multiple heads
  additional_heads: # list of heads to add to predefined arhitecture
    - name: # name of the head (e.g. SegmentationHead) (string)
      params: # params specific to this head (dict)

      # Every head must have its loss function
      loss:
        name: # name of the loss function (e.g. CrossEntropyLoss) (string)
        params: # params specific to this loss function (dict)
```

You can see the list of all currently supported loss functions and their parameters [here](./src/luxonis_train/utils/losses/README.md).

#### Paths

You can specify `pretrained` paths under `model` and `backbone` in one of these formats:

- `local path` (e.g. `path/to/ckpt` or `file://path/to/ckpt`)
- `s3 path` (e.g. `s3://<bucket>/path/to/ckpt`)
- `mlflow path` (e.g. `mlflow://<project_id>/<run_id>/path/to/ckpt` - path is relative to base artifacts path).

#### Advanced configuration

Every head and neck also supports the `attach_index` parameter which specifies on which backbone/neck layer should the head/neck attach to. Layers are indexed from 0 to N where N is the last layer (closest to the head/neck). By default `attach_index` is set to -1 for most of the heads/necks (check [here](./src/luxonis_train/models/heads/README.md)/[here](./src/luxonis_train/models/necks/README.md) for any exceptions) which means that the head/neck attaches to the last previous layer (Python list index convention is used here).

### Trainer

This block configures everything connected to PytrochLightning Trainer object.

```yaml
trainer:
  accelerator: auto # either "cpu" or "gpu, if "auto" then we cauto-selects best available (string)
  devices: auto # either specify how many devices (int) or which specific devices (list) to use. If auto then automatic selection (int|[int]|string)
  strategy: auto # either one of PL stategies or auto (string)
  num_sanity_val_steps: 2 # number of sanity validation steps performed before training (int)
  profiler: null # use PL profiler for GPU/CPU/RAM utilization analysis (string|null)
  verbose: True # print all intermidiate results in console (bool)
```

### Logger

This library uses [LuxonisTrackerPL](https://github.com/luxonis/luxonis-ml/blob/b2399335efa914ef142b1b1a5db52ad90985c539/src/luxonis_ml/ops/tracker.py#L152) for managing different loggers. You can configure it like this:

```yaml
logger:
  project_name: null # name of the project used for logging (string|null)
  project_id: null # id of the project used for logging (relevant if using MLFlow) (string|null)
  run_name: null # name of the run, if empty then auto-generate (string|null)
  run_id: null # id of already create run (relevant if using MLFlow) (string|null)
  save_directory: output # path to the save directory (string)
  is_tensorboard: True # bool if use tensorboard (bool)
  is_wandb: False # bool if use WanDB (bool)
  wandb_entity: null # name of WanDB entity (string|null)
  is_mlflow: False # bool if use MLFlow (bool)
  logged_hyperparams: ["train.epochs", "train.batch_size"] # list of hyperparameters to log (list)
```

### Dataset

To store and load the data we use LuxonisDataset and LuxonisLoader. For specific config parameters refer to [LuxonisML](https://github.com/luxonis/luxonis-ml).

***Note:** Bucket type and storage need to be one of valid Enum values.*

```yaml
dataset:
  dataset_name: null # name of the dataset (string)
  team_id: null # team under which you can find all datasets (string)
  dataset_id: null # id of the dataset (string)
  bucket_type: BucketType.INTERNAL # type of underlying storage (BucketType)
  bucket_storage: BucketStorage.LOCAL # underlying object storage for a bucket (BucketStorage)
  train_view: train # view to use for training (string)
  val_view: val # view to use for validation (string)
  test_view: test # view to use for testing (string)
  json_mode: False # load using JSON annotations instead of MongoDB
```

### Train

Here you can change everything related to actual training of the model.

We use [Albumentations](https://albumentations.ai/docs/) library for `augmentations`. [Here](https://albumentations.ai/docs/api_reference/full_reference/#pixel-level-transforms) you can see a list of all pixel level augmentations supported, and [here](https://albumentations.ai/docs/api_reference/full_reference/#spatial-level-transforms) you see all spatial level transformations. In config you can specify any augmentation from this lists and their params. Additionaly we support `Mosaic4` batch augmentation and letterbox resizing if `keep_aspect_ratio: True`.

For `callbacks` Pytorch Lightning is used and you can check [here](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) for parameter definitions. We also provide some additional callbacks:

- `test_on_finish` - Runs test on the test set with best weights on train finish.
- `export_on_finish` - Runs Exporter when training finishes with parameters set in `export block`. Export weights are automatically set to best weights based on validation loss and if `override_upload_directory` is set then `exporter.upload.upload_directory` is overridden with current MLFlow run (if active). **Note:** You still have to set `exporter.upload.active` to True for this to take effect.
- `upload_checkpoint_on_finish` - Uploads currently best checkpoint (based on validation loss) to the specified remote storage. This can be s3, existing MLFlow, or current MLFlow run (for this use `mlflow://` with no project or run id). See [here](#paths) for examples but note that local path is not valid in this case.

For `optimizers` we use Pytorch library. You can see [here](https://pytorch.org/docs/stable/optim.html) all available optimizers and schedulers.

```yaml
train:
  preprocessing:
    train_image_size: [256, 256] # image size used for training [height, width] (list)
    keep_aspect_ratio: True # bool if keep aspect ration while resizing (bool)
    train_rgb: True # bool if train on rgb or bgr (bool)
    normalize:
      active: True # bool if use normalization (bool)
      params: # params for normalization (dict|null)
    augmentations: # list of Albumentations augmentations
      # - name: Rotate
      #   params:
      #    - limit: 15

  batch_size: 32 # batch size used for trainig (int)
  accumulate_grad_batches: 1 # number of batches for gradient accumulation (int)
  use_weighted_sampler: False # bool if use WeightedRandomSampler for training, only works with classification tasks (bool)
  epochs: 100 # number of training epochs (int)
  num_workers: 2 # number of workers for data loading (int)
  train_metrics_interval: -1 # frequency of computing metrics on train data, -1 if don't perform (int)
  validation_interval: 1 # frequency of computing metrics on validation data (int)
  num_log_images: 4 # maximum number of images to visualize and log (int)
  skip_last_batch: True # bool if skip last batch while training (bool)
  main_head_index: 0 # index of the head which is used for checkpointing based on best metric (int)
  use_rich_text: True # bool if use rich text for console printing

  callbacks: # callback specific parameters (check PL docs)
    test_on_finish: False # bool if should run test when train loop finishes (bool)
    use_device_stats_monitor: False # bool if should use device stats monitor during training (bool)
    export_on_finish: # run export when train loop finishes - takes parameters from export block (bool)
      active: False
      override_upload_directory: True # if active mlflow run then use it as upload directory (if upload active in export block)
    model_checkpoint:
      save_top_k: 3
    upload_checkpoint_on_finish: # uploads best checkpoint based on val loss when train loop finishes
      active: False
      upload_directory: null # either path to s3 or mlflow, if empty mlflow then use current run - should activate mlflow in logger (string)
    early_stopping:
      active: True
      monitor: val_loss/loss
      mode: min
      patience: 5
      verbose: True

  optimizers: # optimizers specific parameters (check Pytorch docs)
    optimizer:
      name: Adam
      params:
    scheduler:
      name: ConstantLR
      params:

  freeze_modules: # defines which modules you want to freeze (not train)
    backbone: False # bool if freeze backbone (bool)
    neck: False # bool if freeze neck (bool)
    heads: [False] # list of bools for specific head freeeze (list[bool])

  losses: # defines weights for losses in multi-head architecture
    log_sub_losses: True # bool if should also log sub-losses (bool)
    weights: [1,1] # list of ints for specific loss weight (list[int])
    # learn_weights: False # bool if weights should be learned (not implemented yet) (bool)
```

### Initialization

You can initialize config object by passing one of:

- Python dictionary
- Path to local or remote .yaml file. The supported formats for paths can be seen [here](#paths)

After that you can create a Config object like:

```python
from luxonis_train.utils.config import Config
cfg = Config(data)
```

***Note**: Config is a singleton object, so once created, you can initialize it without a path/dictionary and access its data. If you want to delete it you can call `Config.clear_instance()`.*

***Note**: If `Config` is initialized with path to an existing MLFlow run and `logger.is_mlflow` is set to True then the logging will continue in this run (meaning no new MLFlow run will be created).*

## Training

Once you've configured your `custom.yaml` file you can train the model using this command:

```
python3 tools/train.py -cfg configs/custom.yaml
```

If you wish to manually override some config parameters you can do this by using `--override` flag. Example of this is:

```
python3 tools/train.py -cfg configs/custom.yaml --override "train.batch_size 8 train.epochs 10"
```

where key and value are space separated and sub-keys are dot(`.`) separated. If structure is of type list then key/sub-key should be a number (e.g. `train.preprocessing.augmentations.0.name RotateCustom`).

## Customizations

We provide a module registry interface through which you can create new backbones, necks, heads, losses, callbacks, optimizers and scheduler. This can be then referenced in the config and used during training. To ensure that everything still works correctly we recommend you inherit from specific base classes:

- Backbone - [BaseBackbone](src/luxonis_train/models/backbones/base_backbone.py)
- Neck - [BaseNeck](src/luxonis_train/models/necks/base_neck.py)
- Head - [BaseHead](src/luxonis_train/models/heads/base_heads.py)
- Loss - [BaseLoss](src/luxonis_train/utils/losses/base_loss.py)
- Callback - [Callback from lightning.pytorch.callbacks](lightning.pytorch.callbacks)
  - This are referenced through Config in `train.callbacks.custom_callbacks` similarly as augmentations (list of dictionaries with module name and params)
- Optimizer - [Optimizer from torch.optim](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
- Scheduler - [LRScheduler from torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

Here is an example of how to use custom loss module:

```python
from luxonis_train.utils.losses.base_loss import BaseLoss
from luxonis_train.utils.registry import LOSSES

@LOSSES.register_module()
class CustomLoss(BaseLoss):
    def forward(self, preds: Any, target: Any, epoch: int, step: int):
      ...
```

And then in the config you reference this CustomLoss under the head.

Lastly there is also an option to run training in a separate thread:

```python
trainer = Trainer(cfg, args_dict)
trainer.train(new_thread=True)
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
    active: True # if should use storage to make study persistant (bool)
    type: local # type of storage, "local" or "remote" (string)
  params: # (key, value) pairs for tunning
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

```
python3 tools/tune.py -cfg configs/custom.yaml
```

## Inference

When you have a trained model you can perform inference with it. To do this setup a path to dataset directory (under `dataset` in config file) and path to local pretrained weights (under `model.pretrained` in config file).

```
python3 tools/infer.py -cfg configs/custom.yaml
```

You can also change on which part of the dataset the inference will run. This is done by defining `inferer` block in the config file.

```yaml
inferer:
  dataset_view: val # view to use for inference (string)
  display: True # bool if should display inference resutls (bool)
  infer_save_directory: null # if this is not null then use this as save directory (string|null)
```

If you want to run Inferer one single image you can do that like this:

```python
inferer = Inferer(args.cfg, args_dict)
inferer.infer_image(img)
```

## Exporting

We support export to `ONNX`, `OpenVINO`, and `DepthAI .blob format` which is used for OAK cameras. By default, we only export to ONNX, but you can also export to the other two formats by changing the config (see below). We are developing a separate tool for model converting which will support even more formats (TBA).

For export you must use the same `model` configuration as in training in addition to `exporter` block in config. In this block you must define `export_weights`, other parameters are optional and can be left as default.

There is also an option to upload .ckpt, .onnx and config.yaml files to remote storage. If this is active you need to specify `upload_directory`. See [here](#paths) for examples but note that local path is not valid in this case. **Note**: This path can be overridden by `train.callbacks.export_on_finish` callback.

```yaml
exporter:
  export_weights: null # path to local weights used for export (string)
  export_save_directory: output_export # path to save directory of exported models (string)
  export_image_size: [256, 256] # image size used for export [height, width] (list)
  export_model_name: model # name of the exported model (string)
  data_type: FP16 # data type used for openVino conversion (string)
  reverse_input_channels: True # bool if reverse input shapes (bool)
  scale_values: [58.395, 57.120, 57.375] # list of scale values (list[int|float])
  mean_values: [123.675, 116.28, 103.53] # list of mean values (list[int|float])
  onnx:
    opset_version: 12 # opset version of onnx used (int)
    dynamic_axes: null # define if dynamic input shapes are used (dict)
  openvino:
    active: False # bool if export to openvino (bool)
  blobconverter:
    active: False # bool if export to blob (bool)
    shaves: 6 # number of shaves used (int)
  upload: # uploads .ckpt, .onnx, config file and modelconverter config file
    active: False  # bool if upload active (bool)
    upload_directory: null # either path to s3 or existing mlflow run (string)
```

Once you have the config file ready you can export the model like this:

```
python3 tools/export.py -cfg configs/custom.yaml
```

## Test Dataset

There is a helper script avaliable used to quickly test the dataset and examine if labels are correct. The script will go over the images in the dataset (validation part) and display them together with all annotations that are present for this particular sample. You must first define `dataset` block in the config and then use it like this:

```
python3 tools/test_dataset.py -cfg configs/custom.yaml
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
