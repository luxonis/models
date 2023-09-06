from .common import *
from .yolov6_loss import YoloV6Loss
from .keypoint_box_loss import KeypointBoxLoss


def init_loss(name, **kwargs):
    """Initializes and returns loss based on provided name and config"""
    return eval(name)(**kwargs)
