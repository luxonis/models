# Luxonis training library
Luxonis training library (luxonis-train) is intended for training deep learning models that can run fast on OAK products. We currently support image classification (multi-class and multi-label), semantic segmentation and object detection tasks. The library also depends on `Luxonis Dataset Format`, which you can see [here](https://github.com/luxonis/luxonis-ml). The main idea is that the user can quickly build a model, that can run on edge devices, by defining model components through config or choose from predefined models, and train it.

The work on this project is in an MVP state, so it may be missing some critical features or have some issues - please report any feedback!

**Table of contents:**
- [Installation](#installation)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Exporting](#exporting)


## Installation:
Since this package relys on `luxonis-ml` library you should first install this as specified [here](https://github.com/luxonis/luxonis-ml/tree/main#installation-and-setup).

## Configuration:
Most of the work is done through a `config.yaml` file, which you can then supply to the Trainer. Config file consists of a few major blocks that are described below. You can create your own config or use/edit one of the already made ones.

### Logger:
This library uses LuxonisTrackerPL for managing different loggers. You can configure it like this: 
```yaml
logger:
  project_name: test_project
  save_directory: output
  is_tensorboard: # (True|False)
  is_wandb: # (True|False)
  is_mlflow: # (True|False)
  wandb_entity: # name of wandb entitiy if is_wandb==True
  mlflow_tracking_uri: # tracking uri if is_mlflow==True
```
### Model:
There are two different ways you can create the model. 

In the first one you defined `backbone`, `neck` and list of `heads`, where you choose each module from a list of supported ones (list of all [backbones](./luxonis_train/models/backbones/README.md), [necks](./luxonis_train/models/necks/README.md) and [heads](./luxonis_train/models/heads/README.md)).

***Note**: Every model must have 1 bacbone and at least 1 head(neck is optional)*

```yaml
model:
  name: test_model
  type: # leave this empty
  pretrained: # local path to weights for whole model (overrides backbone weights)

  backbone:
    name: # neme of the backbone (e.g. ResNet18)
    pretrained: # local path to weights for backbone
    params: # params specific to this backbone

  # neck is optional
  neck: 
    name: # name of the neck (e.g. RepPANNeck)
    params: # params specific to this neck

  # you can define multiple heads
  heads: # first head is most important, its metric used for tracking whole model performance
    - name: # name of the head (e.g. ClasificationHead)
      params: # params specific to this head
      
      # Every head must have its loss function
      loss: 
        name: # name of the loss function (e.g. CrossEntropyLoss)
        params: # params specific to this loss function
```

The second use case is choosing from a list of predefined model arhitectures ([list](./luxonis_train/models/README.md)) and optionally adding additional heads. In this case you define model block in config like this:

```yaml
model:
  name: predefined_model
  type: # name of predefined arhitecture (e.g. YoloV6-n)
  pretrained: # local path to weights for whole model
  params: # model-wise params

  # additional heads are optional, you can define multiple heads
  additional_heads: # list of heads to add to predefined arhitecture
    - name: # name of the head (e.g. SegmentationHead)
      params: # params specific to this head
      
      # Every head must have its loss function
      loss: 
        name: # name of the loss function (e.g. CrossEntropyLoss)
        params: # params specific to this loss function
```

You can see the list of all currently supported loss functions and their parameters [here](./luxonis_train/utils/losses/README.md).

### Dataset
This library is using LuxonisDataset and LuxonisLoader for creating datasets and using them. This can be configured like shown below. Note that both paths are optional but at least one must be set.

```yaml
dataset:
  local_path: # local path to LakeFS repository with webdataset
  s3_path: # endpoint to s3 bucket with data

```

### Train
This block has parameters for configuring training.

***Note:** You can find all possible parameters for early stopping [here](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping).*
```yaml
train:
  image_size: # list of [width, height] for image size used in training (default: [256, 256])
  batch_size: # size of batch used in training (e.g. 8)
  accumulate_grad_batches: # number of batches before doing backward pass (e.g. 2)
  epochs: # number of epochs used (e.g. 20)
  n_workers: # number of workers used for DataLoader (e.g. 3)
  n_metrics: # perform metric evaluation on train set every n epochs (e.g. 5)
  eval_interval: # perform metric evaluation on validation set every n epochs (e.g. 5)  

  # optional block for freezing parts of the model (True means we freeze weights)
  freeze_modules:
    backbone: # (True|False)
    #neck: # (True|False)
    heads: # list of booleans same length as number of heads (e.g. [True, False] for model with 2 heads)

  # optional block for configuring early stopping
  early_stopping:
    monitor: # which value to monitor (e.g. val_loss)
    mode: # (min|max)
```

### Augmentations
We use [Albumentations](https://albumentations.ai/docs/) library for augmentations. [Here](https://albumentations.ai/docs/api_reference/full_reference/#pixel-level-transforms) you can see a list of all pixel level augmentations supported, and [here](https://albumentations.ai/docs/api_reference/full_reference/#spatial-level-transforms) you see all spatial level transformations. In config you can specify any augmentation from this lists and their params. We perform Resize, Normalize and ToTensorV2 by default.

***Note**: If you use spatial transformations check if they are supported for your type of labels.*

```yaml
# you can define multiple augmentations
augmentations:
  - name: # name of the augmentation (e.g. Rotate)
    params: # params of the augmentations specified in Albumentations docs
```

### Optimizer
We support all optimizers from pytorch ([list](https://pytorch.org/docs/stable/optim.html#algorithms)).
```yaml
optimizer:
  name: # name of the optimizer (e.g. Adam)
  params: # params of the optimizer specified in pytorch
```

### Scheduler
We support all schedulers from pytorch ([list](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)).
```yaml
scheduler:
  name: # name of the scheduler (e.g. LinearLR)
  params: # params of the scheduler specified in pytorch
```

## Training
Once you've configured your `custom.yaml` file you can train the model using this command:
```
python3 tools/train.py -cfg configs/custom.yaml
```
You can also specify accelerator with `--accelerator` flag (cpu or gpu) and device with `--devices` that you want to use for training (e.g. `--devices 1`)

If you wish to manually override some config parameters you can do this by using `--override` flag. Example of this is: 
```
python3 tools/train.py -cfg configs/custom.yaml --override "train.batch_size 8 train.epochs 10 train.early_stopping.patience 5"
```
where key and value are space separated and sub-keys are dot(`.`) separated.

## Customize Trainer through API
Before trainig the model you can also additionaly configure it with the use of our [Trainer](./luxonis_train/core/trainer.py) API. Look at [train.py](./tools/train.py) to see how Trainer is initialized. 

From there you can override the loss function for specific head by calling: 
```python
my_loss = my_custom_loss() # this should be torch.nn.Module
trainer = Trainer(args_dict, cfg)
trainer.override_loss(custom_loss=my_loss, head_id=0)
```
You can also override train or validation augmentations like this:
```python
my_train_aug = my_custom_train_aug()
my_val_aug = my_custom_val_aug()
trainer = Trainer(args_dict, cfg)
trainer.override_train_augmentations(aug=my_train_aug)
trainer.override_val_augmentations(aug=my_val_aug)
```
To run training in another thread use this:
```python
trainer = Trainer(args_dict, cfg)
trainer.run(new_thread=True)
```

## Inference
When you have a trained model you can perform inference with it. To do this setup a path to dataset directory (under `dataset` in config file) and path to local pretrained weights (under `model.pretrained` in config file). After that you can can perform inference on whole dataset like this:
```
python3 tools/infer.py -cfg configs/custom_infer.yaml
```

## Exporting
We support export to ONNX, openVINO and .blob format which is used for OAK cameras. For export, you must provide configuration file. This file must include [model](#model) part and `export` part which you can define like this:
```yaml
export:
  weights: # path to pretrained weights of the model
  save_directory: # location where exported files will be saved
  iamge_size: # list of [width, height] for image size used for export (default: [256, 256])
```
Once you have the `custom_export.yaml` config file ready you can export the model like this:
```python
python3 tools/export.py -cfg configs/custom_export.yaml 
```

## Install as a package
If you want to initialize `Trainer` class from anywhere you can install `luxonis-train` as a package like this: 
```
python3 -m pip install -e .
```