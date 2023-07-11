import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, draw_keypoints
from torchvision.ops import box_convert

from luxonis_train.utils.constants import LabelType


def draw_outputs(imgs: torch.Tensor, output: torch.Tensor, head: torch.nn.Module, return_numpy: bool = True,
    unnormalize_img: bool = True, cvt_color: bool = False):
    """Draw model outputs on a batch of images

    Args:
        imgs (torch.Tensor): Batch of images (NCHW format)
        output (torch.Tensor): Model output
        head (torch.nn.Module): Model head used for drawing
        return_numpy (bool, optional): Flag if should return images in numpy format (HWC). Defaults to True.
        unnormalize_img (bool, optional): Unormalize image before drawing to it. Defaults to True.
        cvt_color (bool, optional): Convert from BGR to RGB. Defaults to False.

    Returns:
        list[Union[torch.Tensor, np.ndarray]]: list of images with visualizations 
            (either torch tensors in CHW or numpy arrays in HWC format)
    """

    out_imgs = []
    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        if unnormalize_img:
            curr_img = unnormalize(curr_img, to_uint8=True)
        
        curr_img = head.draw_output_to_img(curr_img, output, i)
        out_imgs.append(curr_img)

    if return_numpy:
        out_imgs = [torch_img_to_numpy(i, cvt_color=cvt_color) for i in out_imgs]

    return out_imgs

def draw_labels(imgs: torch.tensor, label_dict: dict, label_keys: list = None, return_numpy: bool = True,
    unnormalize_img: bool = True, cvt_color: bool = False, overlay: bool = False):
    """Draw all present labels on a batch of images

    Args:
        imgs (torch.tensor): Batch of images (NCHW format)
        label_dict (dict): Dictionary of present labels
        label_keys (list, optional): List of keys for labels to draw, if None use all. Defaults to None
        return_numpy (bool, optional): Flag if should return images in numpy format (HWC). Defaults to True.
        unnormalize_img (bool, optional): Unormalize image before drawing to it. Defaults to True.
        cvt_color (bool, optional): Convert from BGR to RGB. Defaults to False.
        overlay (bool, optional): Draw all labels on the same image. Defaults to False.

    Returns:
        list[Union[torch.Tensor, np.ndarray]]: list of images with visualizations 
            (either torch tensors in CHW or numpy arrays in HWC format)
    """

    _, _, ih, iw = imgs.shape
    out_imgs = []

    if label_keys is None:
        label_keys = list(label_dict.keys())

    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        curr_out_imgs = []
        if unnormalize_img:
            curr_img = unnormalize(curr_img, to_uint8=True)

        for label_key in label_keys:
            if label_key == LabelType.CLASSIFICATION:
                curr_img_class = torch_img_to_numpy(curr_img)
                indices = torch.nonzero(label_dict[label_key][i]).flatten().tolist()
                curr_label_str = ",".join(str(i) for i in indices)
                curr_img_class = cv2.putText(curr_img_class, curr_label_str, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                curr_img_class = numpy_to_torch_img(curr_img_class)
                if overlay:
                    curr_img = curr_img_class
                else:
                    curr_out_imgs.append(curr_img_class)

            if label_key == LabelType.SEGMENTATION:
                curr_label = label_dict[label_key][i]
                masks = curr_label.bool()
                # NOTE: we have to push everything to cpu manually before draw_segmentation_masks (torchvision bug?)
                masks = masks.cpu()
                curr_img = curr_img.cpu()
                curr_img_seg = draw_segmentation_masks(curr_img, masks, alpha=0.4)
                if overlay:
                    curr_img = curr_img_seg
                else:
                    curr_out_imgs.append(curr_img_seg)

            if label_key == LabelType.BOUNDINGBOX:
                curr_label = label_dict[label_key]
                curr_label = curr_label[curr_label[:,0]==i]
                bboxs = box_convert(curr_label[:,2:], "xywh", "xyxy")
                bboxs[:, 0::2] *= iw
                bboxs[:, 1::2] *= ih
                curr_img_bbox = draw_bounding_boxes(curr_img, bboxs)
                if overlay:
                    curr_img = curr_img_bbox
                else:
                    curr_out_imgs.append(curr_img_bbox)

            if label_key == LabelType.KEYPOINT:
                curr_label = label_dict[label_key]
                curr_label = curr_label[curr_label[:,0]==i][:,1:]

                keypoints_flat = torch.reshape(curr_label[:,1:], (-1,3))
                keypoints_points = keypoints_flat[:,:2]
                keypoints_points[:,0] *= iw
                keypoints_points[:,1] *= ih
                keypoints_visibility = keypoints_flat[:,2]

                # torchvision expects format [n_instances, K, 2]
                n_instances = curr_label.shape[0]
                out_keypoints = torch.reshape(keypoints_points, (n_instances, -1, 2)).int()
                curr_img_keypoints = draw_keypoints(curr_img, out_keypoints, colors="red")
                if overlay:
                    curr_img = curr_img_keypoints
                else:
                    curr_out_imgs.append(curr_img_keypoints)

        if overlay:
            curr_out_imgs = [curr_img]

        if return_numpy:
            curr_out_merged = cv2.hconcat(
                [torch_img_to_numpy(i, cvt_color=cvt_color) for i in curr_out_imgs]
            )
        else:
            curr_out_merged = torch.cat(curr_out_imgs, dim=-1) # horizontal concat
        
        out_imgs.append(curr_out_merged)
    return out_imgs


def seg_output_to_bool(data: torch.Tensor, binary_threshold: float = 0.5):
    """ Converts seg head output to 2D boolean mask for visualization"""
    masks = torch.empty_like(data, dtype=torch.bool, device=data.device)
    if data.shape[0] == 1:
        classes = torch.sigmoid(data)
        masks[0] = classes >= binary_threshold
    else:
        classes = torch.argmax(data, dim=0)
        for i in range(masks.shape[0]):
            masks[i] = classes == i
    return masks

def unnormalize(img: torch.Tensor, original_mean: tuple = (0.485, 0.456, 0.406), 
        original_std: tuple = (0.229, 0.224, 0.225), to_uint8: bool = False):
    """ Unnormalizes image back to original values, optionally converts it to uin8"""
    mean = np.array(original_mean)
    std = np.array(original_std)
    new_mean = -mean/std
    new_std = 1/std
    out_img = F.normalize(img, mean=new_mean,std=new_std)
    if to_uint8:
        out_img = torch.clamp(out_img.mul(255), 0, 255).to(torch.uint8)
    return out_img    

def torch_img_to_numpy(img: torch.Tensor, cvt_color: bool = False):
    """ Converts torch image (CHW) to numpy array (HWC). Optionally also converts colors. """
    if img.is_floating_point():
        img = img.mul(255).int()
    img = torch.clamp(img, 0, 255)
    img = np.transpose(img.cpu().numpy().astype(np.uint8), (1, 2, 0))
    img = np.ascontiguousarray(img)
    if cvt_color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def numpy_to_torch_img(img: np.array):
    """ Converts numpy image (HWC) to torch image (CHW) """
    img = torch.from_numpy(img).permute(2,0,1)
    return img
