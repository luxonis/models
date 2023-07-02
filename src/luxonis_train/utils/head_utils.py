import torch
import numpy as np
from luxonis_train.utils.assigners.anchor_generator import generate_anchors
from luxonis_train.utils.boxutils import dist2bbox, non_max_suppression

# utils specific to some head (preprocessing or postprocessing functions)

def yolov6_out2box(output: tuple, head: torch.nn.Module, **kwargs):
    """ Performs post-processing of the YoloV6 output and returns bboxs after NMS"""
    x, cls_score_list, reg_dist_list = output
    anchor_points, stride_tensor = generate_anchors(x, head.stride, 
        head.grid_cell_size, head.grid_cell_offset, is_eval=True)
    pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")

    pred_bboxes *= stride_tensor
    output_merged = torch.cat([
        pred_bboxes, 
        torch.ones((x[-1].shape[0], pred_bboxes.shape[1], 1), dtype=pred_bboxes.dtype, device=pred_bboxes.device), 
        cls_score_list 
    ], axis=-1)

    conf_thres = kwargs.get("conf_thres", 0.001)
    iou_thres = kwargs.get("iou_thres", 0.6)

    output_nms = non_max_suppression(output_merged, conf_thres=conf_thres, iou_thres=iou_thres)

    return output_nms

def generate_blaze_heatmap(keypoints: torch.Tensor, size: int = 128, sigma: int = 2):
    """Generate groundtruth heatmap
        Adapted from: https://github.com/vietanhdev/tf-blazepose/blob/master/src/utils/heatmap.py
    """
    npart = keypoints.shape[0]
    heatmap = torch.zeros(npart, size, size, dtype=torch.float32)
    for i in range(npart):
        curr_keypoints = keypoints[i, 1:]
        for j in range(0,curr_keypoints.shape[0], 3):
            x, y, visibility = curr_keypoints[j:j+3]
            if visibility != 2:
                continue
            pt = [x*size, y*size]
            heatmap[i] = _generate_point_heatmap(
                    heatmap[i], pt, sigma
                )
    return heatmap

def _generate_point_heatmap(img, pt, sigma, type='Gaussian'):
    """Draw label map for 1 point

    Args:
        img: Input image
        pt: Point in format (x, y)
        sigma: Sigma param in Gaussian or Cauchy kernel
        type (str, optional): Type of kernel used to generate heatmap. Defaults to 'Gaussian'.

    Returns:
        np.array: Heatmap image
    """
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = torch.arange(0, size, 1, dtype=torch.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img
