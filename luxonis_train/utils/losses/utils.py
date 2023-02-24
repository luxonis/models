from .common import *
from .yolov6_loss import YoloV6Loss
from .yolov7pose_loss import YoloV7PoseLoss

def get_loss(name, **kwargs):
    return eval(name)(**kwargs)
