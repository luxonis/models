from .common import *
from .bboxyolov6_loss import BboxYoloV6Loss
from .yolov7_pose_loss import YoloV7PoseLoss


def init_loss(name, **kwargs):
    """Initializes and returns loss based on provided name and config"""
    return eval(name)(**kwargs)
