# Luxonis training library
Luxonis training library (luxonis-train) is intended for training deep learning models that can run fast on OAK products. We currently support image classification (multi-class and multi-label), semantic segmentation and object detection tasks. The library also depends on `Luxonis Dataset Format`, which you can see [here](https://github.com/luxonis/luxonis-ml). The main idea is that the user can quickly build a model, that can run on edge devices, by defining model components through config or choose from predefined models, and train it.

The work on this project is in an MVP state, so it may be missing some critical features or have some issues - please report any feedback!

**Table of contents:**
- [Installation](#installation)
  - [Install as a package](#install-as-a-package)
- [Configuration](#configuration)
  - [Model](#model)
  - [Trainer](#trainer)
  - [Logger](#logger)
  - [Dataset](#dataset)
  - [Train](#train)
  - [Initialization](#initialization)
- [Training](#training)
- [Tuning](#tuning)
- [Inference](#inference)
- [Exporting](#exporting)
- [Test Dataset](#test-dataset)


## Installation:
Since this package relys on `luxonis-ml` library you should first install this as specified [here](https://github.com/luxonis/luxonis-ml/tree/main#installation-and-setup).

### Install as a package
If you want to use classes from this library anywhere you can install `luxonis-train` as a package like this:
```
python3 -m pip install -e .
```

## Configuration:
Most of the work is done through a `config.yaml` file, which you must pass as an argument to the Trainer. Config file consists of a few major blocks that are described below. You can create your own config or use/edit one of the already made ones.

### Model:
This is the most important block, that **must be always defined by the user**. There are two different ways you can create the model. 

In the first one you defined `backbone`, `neck` and list of `heads`, where you choose each module from a list of supported ones (list of all [backbones](./luxonis_train/models/backbones/README.md), [necks](./luxonis_train/models/necks/README.md) and [heads](./luxonis_train/models/heads/README.md)). If `n_classes` is not set for the head or is set to `null` then this value will be inherited from the dataset.

***Note**: Every model must have 1 bacbone and at least 1 head (neck is optional).*

```yaml
model:
  name: test_model
  type: # leave this empty (null)
  pretrained: # local path to weights for whole model (overrides backbone weights) (string)

  backbone:
    name: # neme of the backbone (e.g. ResNet18)
    pretrained: # local path to weights for backbone (string)
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

The second use case is choosing from a list of predefined model arhitectures ([list](./luxonis_train/models/README.md)) and optionally adding additional heads. In this case you define model block in config like this:

```yaml
model:
  name: predefined_model
  type: # name of predefined arhitecture (e.g. YoloV6-n) (string)
  pretrained: # local path to weights for whole model (string)
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

You can see the list of all currently supported loss functions and their parameters [here](./luxonis_train/utils/losses/README.md).
#### Advanced configuration:
Every head also supports the `attach_index` parameter which specifies on which backbone/neck layer should the head attach to. Layers are indexed from 0 to N where N is the last layer (closest to the head). By default `attach_index` is set to -1 for all heads which means that the head attaches to the last previous layer (Python list index convention is used here).

### Trainer:
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

### Logger:
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
To store and load the data we use LuxonisDataset and LuxonisLoader. For configuring path to the dataset and othere dataset related parameters use this:

***Note**: At least one of local_path or s3_path parameters must no defined.*

```yaml
dataset:
  team_id: null # team under which you can find all datasets (string)
  dataset_id: null # id of the dataset (string)
  bucket_type: local # underlying storage for images, which can be local or an AWS bucket (local|aws)
  override_bucket_type: False # option to change underlying storage from saved setting in DB (bool)
  train_view: train # view to use for training (string)
  val_view: val # view to use for validation (string)
  test_view: test # view to use for testing (string)
```

### Train
Here you can change everything related to actual training of the model.

We use [Albumentations](https://albumentations.ai/docs/) library for `augmentations`. [Here](https://albumentations.ai/docs/api_reference/full_reference/#pixel-level-transforms) you can see a list of all pixel level augmentations supported, and [here](https://albumentations.ai/docs/api_reference/full_reference/#spatial-level-transforms) you see all spatial level transformations. In config you can specify any augmentation from this lists and their params.

For `callbacks` Pytorch Lightning is used and you can check [here](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) for parameter definitions. We also provide you with two additional callbacks for automatically running test and export on train finish (`test_on_finish` and `export_on_finish` respectively). If `export_on_finish` is used then we will use config from the `exporter` block (explained [here](#exporting)) but for `export_weights` we will use the best checkpoint (based on validation loss) from the current run.


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
    export_on_finish: False # bool if should run export when train loop finishes - should specify config block (bool)
    use_device_stats_monitor: False # bool if should use device stats monitor during training (bool)
    model_checkpoint:
      save_top_k: 3
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
- Path to local .yaml file
- Path to MLFlow artifact with config.json file. This path has to be formated like this: `mlflow://<experiment_id>/<run_id>`
After that you can create a Config object like:
```python
from luxonis_train.utils.config import Config
cfg = Config(data)
```
***Note**: Config is a singleton object, so once created, you can initialize it without a path/dictionary and access its data. If you want to delete it you can call `Config.clear_instance()`.*

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

## Customize Trainer through API
Before trainig the model you can also additionaly configure it with the use of our [Trainer](./luxonis_train/core/trainer.py) API. Look at [train.py](./tools/train.py) to see how Trainer is initialized. 

From there you can override the loss function for specific head by calling: 
```python
my_loss = my_custom_loss() # this should be torch.nn.Module
trainer = Trainer(args_dict, cfg)
trainer.override_loss(custom_loss=my_loss, head_id=0)
```
***Note**: Number of classes (`n_classes`) is beaing passed to each loss initialization by default. In addition to output and labels we also pass current epoch (`epoch`) and current step (`step`) as parameters on every loss function call. If you don't need this use `**kwargs` keyword when defining loss function.*

You can also override train or validation augmentations like this:
```python
my_train_aug = my_custom_train_aug()
my_val_aug = my_custom_val_aug()
trainer = Trainer(cfg, args_dict)
trainer.override_train_augmentations(aug=my_train_aug)
trainer.override_val_augmentations(aug=my_val_aug)
```

To run training in another thread use this:
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
    type: remote # type of storage, "local" or "remote" (string)
  params: # (key, value) pairs for tunning
```

Example of params for tuner block:
```yaml
tuner:
  params:
    train.optimizers.optimizer.name_categorical: ["Adam", "SGD"]
    train.optimizers.optimizer.params.lr_float: [0.0001, 0.001]
    train.batch_size_int: [4, 16, 4]
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
We support export to ONNX, openVINO and .blob format which is used for OAK cameras. By default we only export to ONNX and you should use [modelconverter](TODO) repository for other formats. For export you must use the same `model` configuration as in training in addition to `exporter` block in config. In this block you must define `export_weights`, other parameters are optional and can be left as default.

There is also an option to upload .ckpt, .onnx and config.yaml files to S3 bucket. To do this you have to specify `bucket` and `upload_directory` to configure S3 upload location.

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
  s3_upload:
    active: False # bool if upload .ckpt, .onnx and config file to S3 bucket (bool)
    bucket: null # name of the S3 bucket (string)
    upload_directory: null # location of directory for upload (string)
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