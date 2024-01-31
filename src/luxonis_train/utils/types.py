from typing import Any, Literal

from luxonis_ml.loader import LabelType
from torch import Tensor

Kwargs = dict[str, Any]

OutputTypes = Literal["boxes", "class", "keypoints", "segmentation", "features", "loss"]
Shape = list[int]

ModulePacket = dict[OutputTypes, list[Tensor]]
ShapePacket = dict[OutputTypes, list[Shape]]
Labels = dict[LabelType, Tensor]
