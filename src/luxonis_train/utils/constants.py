from enum import Enum


class LabelType(str, Enum):
    CLASSIFICATION = "class"
    SEGMENTATION = "segmentation"
    BOUNDINGBOX = "bbox"
    KEYPOINT = "keypoints"


class HeadType(Enum):
    CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    SEMANTIC_SEGMENTATION = 3
    INSTANCE_SEGMENTATION = 4
    OBJECT_DETECTION = 5
    KEYPOINT_DETECTION = 6
