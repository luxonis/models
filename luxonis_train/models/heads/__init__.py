from .bisenet_head import BiSeNetHead
from .classification_head import ClassificationHead
from .effide_head import EffiDeHead
from .ikeypoint_head import IKeypoint
from .multilabel_classification_head import MultiLabelClassificationHead
from .segmentation_head import SegmentationHead
from .yolov6_head import YoloV6Head

__all__ = [
    "ClassificationHead",
    "MultiLabelClassificationHead",
    "SegmentationHead",
    "YoloV6Head",
    "EffiDeHead",
    "BiSeNetHead",
    "IKeypoint",
]
