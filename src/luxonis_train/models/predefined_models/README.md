# Predefined models

In addition to definig the model by hand, we offer a list of predefined
models which can be used instead.


## Table Of Content


## SegmentationModel

**Components**

| Name | Alias | Function |
| - | - | - |
| [MicroNet](src/luxonis_train/nodes/README.md#micronet) | segmentation_backbone | Backbone of the model. Can be changed |
| [SegmentationHead](src/luxonis_train/nodes/README.md#segmentationhead) | segmentation_head | Head of the model. |

**Params**

| Key | Type | Default value | Description |
| - | - | - | - |
| task | Literal\["binary", "multiclass", "multilabel"\] | "binary" | Type of the task of the model. |
| backbone | str | "MicroNet"| Name of the node to be used as a backbone. |
| backbone_params | dict | {} | Additional parameters to the backbone. |
| head_params | dict | {} | Additional parameters to the head. |
| loss_params | dict | {} | Additional parameters to the loss. |
| visualizer_params | dict | {} | Additional parameters to the visualizer. |


## DetectionModel

## KeypointDetectionModel

## ClassificationModel
