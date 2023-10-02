from .common import *
from .bboxyolov6_loss import BboxYoloV6Loss
from .keypoint_box_loss import KeypointBoxLoss


def init_loss(name, **kwargs):
    """Initializes and returns loss based on provided name and config"""
    return eval(name)(**kwargs)
