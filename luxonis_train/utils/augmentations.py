import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import torch
from typing import Union

from luxonis_train.utils.config import Config
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *
from luxonis_train.utils.head_utils import generate_blaze_heatmap

class Augmentations:
    def __init__(self):
        """ Base class for creating Augmentations object """
        self.cfg_all = Config()
        self.cfg = self.cfg_all.get("train.preprocessing")

    def _parse_cfg(self, cfg_aug: dict):
        """ Parses provided config and returns Albumentations Compose object"""
        image_size = self.cfg["train_image_size"]
        
        # Always perform Resize
        augmentations = [A.Resize(image_size[0], image_size[1])]
        if cfg_aug:
            for aug in cfg_aug:
                augmentations.append(
                    eval(aug["name"])(**aug.get("params", {}))
                )
        augmentations.append(ToTensorV2())
        
        return A.Compose(
            augmentations, 
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
            keypoint_params=A.KeypointParams(format="xy", label_fields=["keypoints_classes"], remove_invisible=False),
        )

    def __call__(self, data: tuple):
        """ Performs augmentations on provided data"""
        img, anno_dict = data
        present_annotations = anno_dict.keys()
        img_in = img.numpy()

        # albumentations expects with RGB image with HWC format
        img_in = np.transpose(img_in, (1,2,0))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)

        classes = anno_dict.get("class", torch.zeros(1))

        seg = anno_dict.get("segmentation", torch.zeros((1, *img_in.shape[1:])))
        masks = [m.numpy() for m in seg]

        # COCO format in albumentations is [x,y,w,h] non-normalized
        bboxes = anno_dict.get("bbox", torch.zeros((0,5)))
        ih, iw, _ = img_in.shape
        bboxes_points = bboxes[:,1:]
        bboxes_points[:,0::2] *= iw
        bboxes_points[:,1::2] *= ih
        bboxes_points = check_bboxes(bboxes_points)
        bbox_classes = bboxes[:,0]

        # albumentations expects "list" of keypoints e.g. [(x,y),(x,y),(x,y),(x,y)]
        keypoints = anno_dict.get("keypoints", torch.zeros((1,3+1)))
        keypoints_classes = keypoints[:,0]
        keypoints_flat = torch.reshape(keypoints[:,1:], (-1,3))
        keypoints_points = keypoints_flat[:,:2]
        keypoints_points[:,0] *= iw
        keypoints_points[:,1] *= ih
        keypoints_visibility = keypoints_flat[:,2]

        transformed = self.transform(
            image = img_in,
            masks = masks,
            bboxes = bboxes_points,
            bbox_classes = bbox_classes,
            keypoints = keypoints_points,
            keypoints_classes = keypoints_visibility, # not object class, but per-kp class
        )

        final_image, final_bboxes, final_masks, final_keypoints = \
            post_augment_process(transformed, keypoints, 
                keypoints_classes, use_rgb=self.cfg["train_rgb"])

        custom_labels = self.create_custom_labels(final_image, final_bboxes, final_masks, final_keypoints)
        custom_labels = custom_labels[0] if len(custom_labels) else None # NOTE: only supports one custom label per model for now (aka. one head with custom labels)

        out_annotations = create_out_annotations(present_annotations, classes=classes, bboxes=final_bboxes,
            masks=final_masks, keypoints=final_keypoints, custom_labels=custom_labels)

        return final_image, out_annotations

    def create_custom_labels(self, image: np.ndarray, bboxes: torch.Tensor, final_masks: torch.Tensor, final_keypoints: torch.Tensor):
        custom_labels = []
        for head in self.cfg_all.get("model.heads"):
            curr_head_name = head["name"]
            curr_head_type = eval(curr_head_name).type
            if isinstance(curr_head_type, CustomHeadType):
                if "blazeheatmap" in curr_head_name.lower():
                    heatmap = generate_blaze_heatmap(final_keypoints)
                    custom_labels.append(heatmap)
        return custom_labels

class TrainAugmentations(Augmentations):
    def __init__(self):
        """ Class for train augmentations"""
        super().__init__()
        self.transform = self._parse_cfg(
            cfg_aug=self.cfg["augmentations"]
        )

class ValAugmentations(Augmentations):
    def __init__(self):
        """ Class for val augmentations"""
        super().__init__()
        self.transform = self._parse_cfg(
            cfg_aug=[k for k in self.cfg["augmentations"] if k["name"] == "Normalize"]
        )

def post_augment_process(transformed: dict, keypoints: torch.Tensor, keypoints_classes: np.array, use_rgb: bool = True):
    """ Post process augmentation outputs to prepare for training """
    transformed_image = transformed["image"]
    if not use_rgb:
        transformed_image = transformed_image.flip(-3)

    transformed_mask = torch.stack(transformed["masks"]) # stack list of masks

    transformed_bboxes = torch.tensor(transformed["bboxes"])
    transformed_bbox_classes = torch.tensor(transformed["bbox_classes"])
    # merge bboxes and classes back together
    transformed_bbox_classes = torch.unsqueeze(transformed_bbox_classes, dim=-1)
    out_bboxes = torch.cat((transformed_bbox_classes, transformed_bboxes), dim=1)

    out_bboxes[:,1::2] /= transformed_image.shape[2]
    out_bboxes[:,2::2] /= transformed_image.shape[1]

    transformed_keypoints = torch.tensor(transformed["keypoints"])
    transformed_keypoints_classes = torch.tensor(transformed["keypoints_classes"])
    # merge keypoints and classes back together
    transformed_keypoints_classes = torch.unsqueeze(transformed_keypoints_classes, dim=-1)
    out_keypoints = torch.cat((transformed_keypoints, transformed_keypoints_classes), dim=1)

    out_keypoints = torch.reshape(out_keypoints, (-1, 3))
    out_keypoints = mark_invisible_keypoints(out_keypoints, transformed_image)
    out_keypoints[...,0] /= transformed_image.shape[2]
    out_keypoints[...,1] /= transformed_image.shape[1]
    out_keypoints = torch.reshape(out_keypoints, (keypoints.shape[0], keypoints.shape[1]-1))

    final_keypoints = torch.zeros_like(keypoints)
    final_keypoints[:,1:] = out_keypoints
    final_keypoints[:,0] = keypoints_classes

    return transformed_image, out_bboxes, transformed_mask, final_keypoints

def mark_invisible_keypoints(keypoints: torch.Tensor, image: np.array):
    """ Mark invisible keypoints with label == 0 """
    _, h, w = image.shape
    for kp in keypoints:
        if not(0<=kp[0]<w and 0<=kp[1]<h):
            kp[2] = 0
    return keypoints

def check_bboxes(bboxes: torch.Tensor):
    """ Check bbox annotations and correct those with widht or height 0 """
    for i in range(bboxes.shape[0]):
        if bboxes[i, 2] == 0:
            bboxes[i, 2] = 1
        if bboxes[i, 3] == 0:
            bboxes[i, 3] = 1
    return bboxes

def create_out_annotations(present_annotations: list, classes: torch.Tensor, bboxes: torch.Tensor,
    masks: torch.Tensor, keypoints: torch.Tensor, custom_labels: Union[None, torch.Tensor]):
    """ Create dictionary of output annotations """
    out_annotations = {}
    if "class" in present_annotations:
        out_annotations["class"] = classes
    if "bbox" in present_annotations:
        out_annotations["bbox"] = bboxes
    if "segmentation" in present_annotations:
        out_annotations["segmentation"] = masks
    if "keypoints" in present_annotations:
        out_annotations["keypoints"] = keypoints
    if custom_labels:
        out_annotations["custom"] = custom_labels
    return out_annotations
