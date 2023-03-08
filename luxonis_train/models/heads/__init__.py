from .classification_head import ClassificationHead
from .multilabel_classification_head import MultiLabelClassificationHead
from .segmentation_head import SegmentationHead
from .yolov6_head import YoloV6Head
from .effide_head import EffiDeHead

__all__ = [
    "ClassificationHead",
    "MultiLabelClassificationHead",
    "SegmentationHead",
    "YoloV6Head",
    "EffiDeHead"
]
