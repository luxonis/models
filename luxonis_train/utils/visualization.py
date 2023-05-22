import torch
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, draw_keypoints
import torchvision.transforms.functional as F

from luxonis_train.utils.head_type import *
from luxonis_train.models.heads import *
from luxonis_train.utils.boxutils import xywh2xyxy_coco
from luxonis_train.utils.head_utils import yolov6_out2box


def draw_on_images(imgs: torch.Tensor, data: torch.Tensor, head: torch.nn.Module,
    is_label: bool = False, return_numpy: bool = True, **kwargs):
    """ Draws labels and model outputs to whole batch of images

    Args:
        imgs (torch.Tensor): batch of images (NCHW format)
        data (torch.Tensor): output or label data
        head (torch.nn.Module): head used for generating outputs 
        is_label (bool, optional): flag if visualizing labels. Defaults to False.
        return_numpy (bool, optional): flag if should return images in numpy format (HWC). Defaults to True.

    Returns:
        list[Union[torch.Tensor, np.ndarray]]: list of images with visualizations 
            (either torch tensors in CHW or numpy arrays in HWC format)
    """

    out_imgs = []
    unormalize_images = kwargs.get("unormalize", True)
    
    if isinstance(head, YoloV6Head) and not is_label:
        if "conf_thres" not in kwargs:
            kwargs["conf_thres"] = 0.3
        if "iou_thres" not in kwargs:
            kwargs["iou_thres"] = 0.6
        data = yolov6_out2box(data, head, **kwargs)

    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        if unormalize_images:
            curr_img = unnormalize(curr_img, to_uint8=True)

        # special cases to get labels of current image for object and keypoint detection
        if isinstance(head.type, ObjectDetection) and is_label:
            curr_data = data[data[:,0]==i]
        elif isinstance(head.type, KeyPointDetection) and is_label:
            curr_data = data[data[:,0]==i][:,1:]
        else:
            curr_data = data[i]

        curr_img = _draw_on_image(curr_img, curr_data, head, is_label, **kwargs)
        out_imgs.append(curr_img)

    if return_numpy:
        cvt_color = kwargs.get("cvt_color", False)
        out_imgs = [torch_img_to_numpy(i, cvt_color=cvt_color) for i in out_imgs]

    return out_imgs

def draw_only_labels(imgs: torch.tensor, anno_dict: dict, image_size: tuple, 
    return_numpy: bool = True, **kwargs):
    """ Draw all present labels on batch on images

    Args:
        imgs (torch.tensor): batch of images (NCHW format)
        anno_dict (dict): dictionary of present labels
        image_size (tuple): tuple of image size ([W, H])
        return_numpy (bool, optional): flag if should return images in numpy format (HWC). Defaults to True.

    Returns:
        list[Union[torch.Tensor, np.ndarray]]: list of images with visualizations 
            (either torch tensors in CHW or numpy arrays in HWC format)
    """
    iw, ih = image_size
    out_imgs = []
    unormalize_images = kwargs.get("unormalize", True)
    cvt_color = kwargs.get("cvt_color", False)

    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        curr_out_imgs = []
        if unormalize_images:
            curr_img = unnormalize(curr_img, to_uint8=True)
        
        for label_type in anno_dict:
            if label_type == "class":
                curr_img_class = torch_img_to_numpy(curr_img)
                curr_label = int(anno_dict[label_type][i])
                curr_img_class = cv2.putText(curr_img_class, f"{curr_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                curr_img_class = numpy_to_torch_img(curr_img_class)
                curr_out_imgs.append(curr_img_class)

            if label_type == "segmentation":
                curr_label = anno_dict[label_type][i]
                masks = curr_label.bool()
                curr_img_seg = draw_segmentation_masks(curr_img, masks, alpha=0.4)
                curr_out_imgs.append(curr_img_seg)

            if label_type == "bbox":
                curr_label = anno_dict[label_type]
                curr_label = curr_label[curr_label[:,0]==i]
                bboxs = xywh2xyxy_coco(curr_label[:, 2:])
                bboxs[:, 0::2] *= iw
                bboxs[:, 1::2] *= ih
                curr_img_bbox = draw_bounding_boxes(curr_img, bboxs)
                curr_out_imgs.append(curr_img_bbox)

            if label_type == "keypoints":
                curr_label = anno_dict[label_type]
                curr_label = curr_label[curr_label[:,0]==i][:,1:]

                keypoints_flat = torch.reshape(curr_label[:,1:], (-1,3))
                keypoints_points = keypoints_flat[:,:2]
                keypoints_points[:,0] *= iw
                keypoints_points[:,1] *= ih
                keypoints_visibility = keypoints_flat[:,2]
                
                # torchvision expects format [num_instances, K, 2]
                out_keypoints = torch.reshape(keypoints_points, (-1, 17, 2)).int()
                curr_img_keypoints = draw_keypoints(curr_img, out_keypoints, colors="red")
                curr_out_imgs.append(curr_img_keypoints)

        if return_numpy:
            curr_out_merged = cv2.hconcat(
                [torch_img_to_numpy(i, cvt_color=cvt_color) for i in curr_out_imgs]
            )
        else:
            curr_out_merged = torch.cat(curr_out_imgs, dim=-1) # horizontal concat
        
        out_imgs.append(curr_out_merged)
    return out_imgs


# TODO: find better colormap for colors and put it into torchvision draw functions
def _draw_on_image(img: torch.Tensor, data: torch.Tensor, head: torch.nn.Module, is_label: bool = False, **kwargs):
    """ Draws model output/labels on single image based on head type """
    img_shape = head.original_in_shape[2:]

    if isinstance(head.type, Classification):
        # convert image to cv2 to add text and then convert back to torch img
        # TODO: find if there is a better way to add text to torch img
        img = torch_img_to_numpy(img)
        curr_class = data.argmax()
        img = cv2.putText(img, f"{curr_class}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        img = numpy_to_torch_img(img)
        return img

    elif isinstance(head.type, MultiLabelClassification):
        # TODO: what do we want to visualize in this case?
        raise NotImplementedError()

    elif isinstance(head.type, SemanticSegmentation):
        if is_label:
            masks = data.bool()
        else:
            masks = seg_output_to_bool(data)
        # push both inputs to function to cpu to get it working, probably a bug in torchvision
        img = img.cpu()
        masks = masks.cpu()
        img = draw_segmentation_masks(img, masks, alpha=0.4)
        return img

    elif isinstance(head.type, ObjectDetection):
        label_map = kwargs.get("label_map", None)

        if isinstance(head, YoloV6Head):
            if is_label:
                bboxs = xywh2xyxy_coco(data[:, 2:])
                bboxs[:, 0::2] *= img_shape[1]
                bboxs[:, 1::2] *= img_shape[0]
                labels = data[:,1].int()
            else:
                bboxs = data[:,:4]
                labels = data[:,5].int()

            if label_map:
                labels_list = [label_map[i] for i in labels]
            
            img = draw_bounding_boxes(img, bboxs, labels=labels_list if label_map else None)
            return img

    elif isinstance(head.type, KeyPointDetection):
        if is_label:
            keypoints_flat = torch.reshape(data[:,1:], (-1,3))
            keypoints_points = keypoints_flat[:,:2]
            keypoints_points[:,0] *= img_shape[1]
            keypoints_points[:,1] *= img_shape[0]
            #TODO: do we want to visualize based on visibility? now it just draws all (false point at 0,0 for invisible points)
            keypoints_visibility = keypoints_flat[:,2]

            # torchvision expects format [num_instances, K, 2]
            out_keypoints = torch.reshape(keypoints_points, (-1, 17, 2)).int()
            img = draw_keypoints(img, out_keypoints, colors="red")
            return img
        else:
            # TODO: need to implement this but don't have a keypoint yet to check what is the output from it
            # probably have to do something very similar than for labels
            raise NotImplementedError()

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
