import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.ops import box_convert
from torchvision.utils import (
    draw_bounding_boxes,
    draw_keypoints,
    draw_segmentation_masks,
)


def preprocess_image(
    imgs: Tensor,
    unnormalize_img: bool = True,
    normalize_params: dict[str, list[float]] | None = None,
) -> list[Tensor]:
    """Draw model outputs on a batch of images

    Args:
            imgs (Tensor): Batch of images (NCHW format)
            output (Tensor): Model output
            head (torch.nn.Module): Model head used for drawing
            return_numpy (bool, optional): Flag if should return images in
            numpy format (HWC). Defaults to True.
            unnormalize_img (bool, optional): Unormalize image before drawing to it.
            Defaults to True.
            cvt_color (bool, optional): Convert from BGR to RGB. Defaults to False.
            normalize_params (Optional[Dict[str, List[float]]], optional):
            Params used for normalization. Defaults to None.

    Returns:
            list[Union[Tensor, np.ndarray]]: List of images with visualizations
                    (either torch tensors in CHW or numpy arrays in HWC format)
    """

    out_imgs = []
    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        if unnormalize_img:
            curr_img = unnormalize(
                curr_img, to_uint8=True, normalize_params=normalize_params
            )
        else:
            curr_img = curr_img.to(torch.uint8)

        out_imgs.append(curr_img)

    return out_imgs


def draw_classification_labels(img: Tensor, label: Tensor) -> Tensor:
    curr_img = torch_img_to_numpy(img)
    indices = torch.nonzero(label).flatten().tolist()
    curr_label_str = ",".join(str(i) for i in indices)
    curr_img_class = cv2.putText(
        curr_img,
        curr_label_str,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return numpy_to_torch_img(curr_img_class)


def draw_segmentation_labels(img: Tensor, label: Tensor) -> Tensor:
    masks = label.bool()
    masks = masks.cpu()
    img = img.cpu()
    return draw_segmentation_masks(img, masks, alpha=0.4)


def draw_bounding_box_labels(img: Tensor, label: Tensor) -> Tensor:
    _, H, W = img.shape
    bboxs = box_convert(label[:, 2:], "xywh", "xyxy")
    bboxs[:, 0::2] *= W
    bboxs[:, 1::2] *= H
    return draw_bounding_boxes(img, bboxs)


def draw_keypoint_labels(img: Tensor, label: Tensor) -> Tensor:
    _, H, W = img.shape
    keypoints_unflat = label[:, 1:].reshape(-1, 3)
    keypoints_points = keypoints_unflat[:, :2]
    keypoints_points[:, 0] *= W
    keypoints_points[:, 1] *= H

    n_instances = label.shape[0]
    if n_instances == 0:
        out_keypoints = keypoints_points.reshape((-1, 2)).unsqueeze(0).int()
    else:
        out_keypoints = keypoints_points.reshape((n_instances, -1, 2)).int()

    return draw_keypoints(img, out_keypoints, colors="red")


def seg_output_to_bool(data: Tensor, binary_threshold: float = 0.5) -> Tensor:
    """Converts seg head output to 2D boolean mask for visualization"""
    masks = torch.empty_like(data, dtype=torch.bool, device=data.device)
    if data.shape[0] == 1:
        classes = torch.sigmoid(data)
        masks[0] = classes >= binary_threshold
    else:
        classes = torch.argmax(data, dim=0)
        for i in range(masks.shape[0]):
            masks[i] = classes == i
    return masks


def unnormalize(
    img: Tensor,
    normalize_params: dict[str, list[float]] | None = None,
    to_uint8: bool = False,
) -> Tensor:
    """Unnormalizes image back to original values, optionally converts it to uin8"""
    normalize_params = normalize_params or {}
    mean = np.array(normalize_params.get("mean", [0.485, 0.456, 0.406]))
    std = np.array(normalize_params.get("std", [0.229, 0.224, 0.225]))
    new_mean = -mean / std
    new_std = 1 / std
    out_img = F.normalize(img, mean=new_mean, std=new_std)
    if to_uint8:
        out_img = torch.clamp(out_img.mul(255), 0, 255).to(torch.uint8)
    return out_img


def torch_img_to_numpy(img: Tensor, cvt_color: bool = False) -> np.ndarray:
    """Converts torch image (CHW) to numpy array (HWC).
    Optionally also converts colors."""
    if img.is_floating_point():
        img = img.mul(255).int()
    img = torch.clamp(img, 0, 255)
    img = np.transpose(img.cpu().numpy().astype(np.uint8), (1, 2, 0))
    img = np.ascontiguousarray(img)
    if cvt_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def numpy_to_torch_img(img: np.array) -> Tensor:
    """Converts numpy image (HWC) to torch image (CHW)"""
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img
