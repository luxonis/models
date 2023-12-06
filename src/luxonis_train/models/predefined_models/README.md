# Predefined models

In addition to definig the model by hand, we offer a list of predefined
models which can be used instead.

## Table Of Content

- [SegmentationModel](#segmentationmodel)
- [DetectionModel](#detectionmodel)
- [KeypointDetectionModel](#keypointdetectionmodel)
- [ClassificationModel](#classificationmodel)

## SegmentationModel

**Components**

| Name                                                                           | Alias                 | Function                                                                |
| ------------------------------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------- |
| [MicroNet](../../nodes/README.md#micronet)                                     | segmentation_backbone | Backbone of the model. Can be changed                                   |
| [SegmentationHead](../../nodes/README.md#segmentationhead)                     | segmentation_head     | Head of the model.                                                      |
| [BCEWithLogitsLoss](../../attached_modules/losses/README.md#bcewithlogitsloss) | segmentation_loss     | Loss of the model when the task is set to "binary".                     |
| [CrossEntropyLoss](../../attached_modules/losses/README.md#crossentropyloss)   | segmentation_loss     | Loss of the model when the task is set to "multiclass" or "multilabel". |
| [JaccardIndex](../../attached_modules/metrics/README.md#torchmetrics) | segmentation_jaccard_index     | Main metric of the model.                     |
| [F1Score](../../attached_modules/metrics/README.md#torchmetrics) | segmentation_f1_score     | Secondary metric of the model.                     |
| [SegmentationVisualizer](../../attached_modules/visualizers/README.md#segmentationvisualizer) | segmentation_visualizer    | Visualizer of the `SegmentationHead`.                     |

**Params**

| Key               | Type                                            | Default value | Description                                |
| ----------------- | ----------------------------------------------- | ------------- | ------------------------------------------ |
| task              | Literal\["binary", "multiclass"\] | "binary"      | Type of the task of the model.             |
| backbone          | str                                             | "MicroNet"    | Name of the node to be used as a backbone. |
| backbone_params   | dict                                            | {}            | Additional parameters to the backbone.     |
| head_params       | dict                                            | {}            | Additional parameters to the head.         |
| loss_params       | dict                                            | {}            | Additional parameters to the loss.         |
| visualizer_params | dict                                            | {}            | Additional parameters to the visualizer.   |

## DetectionModel

**Components**

| Name                                                                           | Alias                 | Function                                                                |
| ------------------------------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------- |
| [EfficientRep](../../nodes/README.md#efficientrep)                                     | detection_backbone | Backbone of the model.                                   |
| [RepPANNeck](../../nodes/README.md#reppanneck)                                     | detection_neck | Neck of the model.                                  |
| [EfficientBBoxHead](../../nodes/README.md#efficientbboxhead)                     | detection_head     | Head of the model.                                                      |
| [AdaptiveDetectionLoss](../../attached_modules/losses/README.md#adaptivedetectionloss) | detection_loss     | Loss of the model.                     |
| [MeanAveragePrecision](../../attached_modules/metrics/README.md#meanaverageprecision) | detection_map     | Main metric of the model.                     |
| [BBoxVisualizer](../../attached_modules/visualizers/README.md#bboxvisualizer) | detection_visualizer    | Visualizer of the `detection_head`.                     |

**Params**

| Key               | Type                                            | Default value | Description                                |
| ----------------- | ----------------------------------------------- | ------------- | ------------------------------------------ |
| use_neck              | bool | True      | Whether to include the neck in the model.             |
| backbone_params   | dict                                            | {}            | Additional parameters to the backbone.     |
| neck_params   | dict                                            | {}            | Additional parameters to the neck.     |
| head_params       | dict                                            | {}            | Additional parameters to the head.         |
| loss_params       | dict                                            | {}            | Additional parameters to the loss.         |
| visualizer_params | dict                                            | {}            | Additional parameters to the visualizer.   |

## KeypointDetectionModel

**Components**

| Name                                                                           | Alias                 | Function                                                                |
| ------------------------------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------- |
| [EfficientRep](../../nodes/README.md#efficientrep)                                     | kpt_detection_backbone | Backbone of the model.                                   |
| [RepPANNeck](../../nodes/README.md#reppanneck)                                     | kpt_detection_neck | Neck of the model.                                  |
| [ImplicitKeypointBBoxHead](../../nodes/README.md#implicitkeypointbboxhead)                     | kpt_detection_head     | Head of the model.                                                      |
| [ImplicitKeypointBBoxLoss](../../attached_modules/losses/README.md#implicitkeypointbboxloss) | kpt_detection_loss     | Loss of the model.                     |
| [ObjectKeypointSimilarity](../../attached_modules/metrics/README.md#objectkeypointsimilarity) | kpt_detection_oks     | Main metric of the model.                     |
| [MeanAveragePrecisionKeypoints](../../attached_modules/metrics/README.md#meanaverageprecisionkeypoints) | kpt_detection_map     | Secondary metric of the model.                     |
| [BBoxVisualizer](../../attached_modules/visualizers/README.md#bboxvisualizer) |    | Visualizer for bounding boxes. Combined with keypoint visualizer in [MultiVisualizer](../../attached_modules/visualizers/README.md#multivisualizer).       |
| [KeypointVisualizer](../../attached_modules/visualizers/README.md#keypointvisualizer) |    | Visualizer for keypoints. Combined with keypoint visualizer in [MultiVisualizer](../../attached_modules/visualizers/README.md#multivisualizer) |

**Params**

| Key               | Type                                            | Default value | Description                                |
| ----------------- | ----------------------------------------------- | ------------- | ------------------------------------------ |
| use_neck              | bool | True      | Whether to include the neck in the model.             |
| backbone_params   | dict                                            | {}            | Additional parameters to the backbone.     |
| neck_params   | dict                                            | {}            | Additional parameters to the neck.     |
| head_params       | dict                                            | {}            | Additional parameters to the head.         |
| loss_params       | dict                                            | {}            | Additional parameters to the loss.         |
| kpt_visualizer_params | dict                                            | {}            | Additional parameters to the keypoint visualizer.   |
| bbox_visualizer_params | dict                                            | {}            | Additional parameters to the bbox visualizer.   |

## ClassificationModel

Basic model for classification. Can be used for multiclass and multilabel tasks.

**Components**

| Name                                                                           | Alias                 | Function                                                                |
| ------------------------------------------------------------------------------ | --------------------- | ----------------------------------------------------------------------- |
| [MicroNet](../../nodes/README.md#micronet)                                     | classification_backbone | Backbone of the model. Can be changed                                   |
| [ClassificationHead](../../nodes/README.md#classificationhead)                     | classification_head     | Head of the model.                                                      |
| [CrossEntropyLoss](../../attached_modules/losses/README.md#crossentropyloss)   | classification_loss     | Loss of the model. |
| [F1Score](../../attached_modules/metrics/README.md#torchmetrics) | classification_f1_score     | Main metric of the model.                     |
| [Accuracy](../../attached_modules/metrics/README.md#torchmetrics) | classification_accuracy     | Secondary metric of the model.                     |
| [Recall](../../attached_modules/metrics/README.md#torchmetrics) | classification_recall     | Secondary metric of the model.                     |

**Params**

| Key               | Type                                            | Default value | Description                                |
| ----------------- | ----------------------------------------------- | ------------- | ------------------------------------------ |
| task              | Literal\["multiclass", "multilabel"\] | "multiclass"      | Type of the task of the model.             |
| backbone          | str                                             | "MicroNet"    | Name of the node to be used as a backbone. |
| backbone_params   | dict                                            | {}            | Additional parameters to the backbone.     |
| head_params       | dict                                            | {}            | Additional parameters to the head.         |
| loss_params       | dict                                            | {}            | Additional parameters to the loss.         |
| visualizer_params | dict                                            | {}            | Additional parameters to the visualizer.   |
