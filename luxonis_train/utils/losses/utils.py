from .common import *
from .yolov6_loss import YoloV6Loss 

def get_loss(name, **kwargs):
    return eval(name)(**kwargs)