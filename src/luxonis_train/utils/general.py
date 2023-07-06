import math
import torch
from luxonis_train.utils.head_type import *

def make_divisible(x: int, divisor: int):
    """ Upward revision the value x to make it evenly divisible by the divisor. """
    return math.ceil(x / divisor) * divisor

def dummy_input_run(module: torch.nn.Module, input_shape: list, multi_input: bool = False):
    """ Runs dummy input through the module"""
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


def get_head_name(head: torch.nn.Module, idx: int):
    " Returns generated head name based on its class and id """
    return head.__class__.__name__ + f"_{idx}"

def get_current_label(head_type: object, labels: dict):
    """ Returns the right type of labels depending on head type """
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
        
        # TODO: this works for this specific keypoint head but might not be general enough
        from luxonis_train.utils.boxutils import xywh2cxcywh
        kpts = labels["keypoints"]
        boxes = labels["bbox"]
        nkpts = (kpts.shape[1] - 2) // 3
        targets = torch.zeros((len(boxes), nkpts * 2 + 6))
        targets[:, :2] = boxes[:, :2]
        targets[:, 2:6] = xywh2cxcywh(boxes[:, 2:])
        targets[:,6::2] = kpts[:,2::3] # insert kp x coordinates
        targets[:,7::2] = kpts[:,3::3] # insert kp y coordinates
        #return labels["keypoints"]
        return targets
    else:
        raise RuntimeError(f"No labels for head type {head_type}.")