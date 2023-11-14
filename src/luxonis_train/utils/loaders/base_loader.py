import torch
from abc import ABC, abstractmethod
from torch import Tensor, FloatTensor
from typing import Dict, Tuple, List
from luxonis_ml.data import LabelType


Labels = Dict[LabelType, Tensor]
LuxonisLoaderTorchOutput = Tuple[Tensor, Labels]


class BaseLoaderTorch(ABC):
    """Base abstract loader class that enforces LuxonisLoaderTorchOutput output label structure."""

    @abstractmethod
    def __len__(self) -> int:
        """Returns length of the dataset"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        """Loads sample from dataset

        Args:
            idx (int): Sample index

        Returns:
            LuxonisLoaderTorchOutput: Sample's data in LuxonisLoaderTorchOutput format
        """
        pass


def collate_fn(batch: List[LuxonisLoaderTorchOutput]) -> Tuple[FloatTensor, Dict]:
    """Default collate function used for training

    Args:
        batch (list): List of images and their annotations in LuxonisLoaderOutput format

    Returns:
        Tuple[FloatTensor, Dict]:
            imgs: Tensor of images (torch.float32) of shape [N, 3, H, W]
            out_annotations: Dictionary with annotations
                {
                    LabelType.CLASSIFICATION: Tensor of shape [N, classes] with value 1 for present class
                    LabelType.SEGMENTATION: Tensor of shape [N, classes, H, W] with value 1 for pixels that are part of the class
                    LabelType.BOUNDINGBOX: Tensor of shape [instances, 6] with [image_id, class, x_min_norm, y_min_norm, w_norm, h_norm]
                    LabelType.KEYPOINT: Tensor of shape [instances, n_keypoints*3] with [image_id, x1_norm, y1_norm, vis1, x2_norm, y2_norm, vis2, ...]
                }
    """

    zipped = zip(*batch)
    imgs, anno_dicts = zipped
    imgs = torch.stack(imgs, 0)

    present_annotations = anno_dicts[0].keys()
    out_annotations = {anno: None for anno in present_annotations}

    if LabelType.CLASSIFICATION in present_annotations:
        class_annos = [anno[LabelType.CLASSIFICATION] for anno in anno_dicts]
        out_annotations[LabelType.CLASSIFICATION] = torch.stack(class_annos, 0)

    if LabelType.SEGMENTATION in present_annotations:
        seg_annos = [anno[LabelType.SEGMENTATION] for anno in anno_dicts]
        out_annotations[LabelType.SEGMENTATION] = torch.stack(seg_annos, 0)

    if LabelType.BOUNDINGBOX in present_annotations:
        bbox_annos = [anno[LabelType.BOUNDINGBOX] for anno in anno_dicts]
        label_box = []
        for i, box in enumerate(bbox_annos):
            l_box = torch.zeros((box.shape[0], 6))
            l_box[:, 0] = i  # add target image index for build_targets()
            l_box[:, 1:] = box
            label_box.append(l_box)
        out_annotations[LabelType.BOUNDINGBOX] = torch.cat(label_box, 0)

    if LabelType.KEYPOINT in present_annotations:
        keypoint_annos = [anno[LabelType.KEYPOINT] for anno in anno_dicts]
        label_keypoints = []
        for i, points in enumerate(keypoint_annos):
            l_kps = torch.zeros((points.shape[0], points.shape[1] + 1))
            l_kps[:, 0] = i  # add target image index for build_targets()
            l_kps[:, 1:] = points
            label_keypoints.append(l_kps)
        out_annotations[LabelType.KEYPOINT] = torch.cat(label_keypoints, 0)

    return imgs, out_annotations
