import math
import torch
from luxonis_train.utils.head_type import *

def make_divisible(x, divisor):
    """ Upward revision the value x to make it evenly divisible by the divisor. """
    return math.ceil(x / divisor) * divisor

def dummy_input_run(module, input_shape, multi_input=False):
    """ Run dummy input through the module"""
    module.eval()
    if multi_input:
        input = [torch.zeros(i) for i in input_shape]
    else:
        input = torch.zeros(input_shape)
    
    out = module(input)
    module.train()
    if isinstance(out,list):
        shapes = []
        for o in out:
            shapes.append(list(o.shape))
        return shapes
    else:
        return [list(out.shape)]
    
def get_head_name(head, idx):
    " Return generated head name based on its class and id """
    return head.__class__.__name__ + f"_{idx}"

def get_current_label(head_type, labels):
    """ Return the right type of labels depending on head type """
    present_annotations = labels.keys()
    if isinstance(head_type, Classification) or isinstance(head_type, MultiLabelClassification):
        if "class" not in present_annotations:
            raise RuntimeError("Class labels not avaliable but needed for training.")
        return labels["class"]
    elif isinstance(head_type, SemanticSegmentation) or isinstance(head_type, InstanceSegmentation):
        if "segmentation" not in present_annotations:
            raise RuntimeError("Segmentation labels not avaliable but needed for training.")
        return labels["segmentation"]
    elif isinstance(head_type, ObjectDetection):
        if "bbox" not in present_annotations:
            raise RuntimeError("Bbox labels not avaliable but needed for training.")
        return labels["bbox"]
    elif isinstance(head_type, KeyPointDetection):
        if "keypoints" not in present_annotations:
            raise RuntimeError("Keypoints labels not avaliable but needed for training.")
        return labels["keypoints"]
    else:
        raise RuntimeError(f"No labels for head type {head_type}.")
