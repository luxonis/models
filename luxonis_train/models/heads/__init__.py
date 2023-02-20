from .classification_head import ClassificationHead
from .segmentation_head import SegmentationHead
from .yolov6_head import YoloV6Head
from .effide_head import EffiDeHead

__all__ = [
    "ClassificationHead",
    "SegmentationHead",
    "YoloV6Head",
    "EffiDeHead"
]
