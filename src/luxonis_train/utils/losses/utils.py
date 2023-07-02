from .common import *
from .yolov6_loss import YoloV6Loss 

def init_loss(name, **kwargs):
    """ Initializes and returns loss based on provided name and config"""
    return eval(name)(**kwargs)