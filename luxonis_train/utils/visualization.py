import torch
import cv2
import numpy as np
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks, draw_keypoints
import torchvision.transforms.functional as F

from luxonis_train.utils.head_type import *
from luxonis_train.models.heads import *
from luxonis_train.utils.boxutils import xywh2xyxy_coco
from luxonis_train.utils.head_utils import yolov6_out2box


# TODO: find better colormap for colors and put it into torchvision draw functions
def draw_on_image(img, data, head, is_label=False, **kwargs):
    img_shape = head.original_in_shape[2:]

    if isinstance(head.type, Classification):
        # resize image
        width_orig, height_orig, _ = img.shape
        width_new = 252 # fix image width to 252
        height_new = int(height_orig * width_new/width_orig)
        img = cv2.resize(img, (width_new,height_new))
        # construct info box
        info_box = np.zeros((40, 252, 3),dtype=np.uint8)
        info_box[True] = 255 #change all values to 255
        info_text = f"idx: {data['label']}; prediction: {data['prediction']}"
        cv2.putText(info_box,info_text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        # join resized image and infobox
        return np.concatenate((img, info_box), axis=0)
    elif isinstance(head.type, MultiLabelClassification):
        # TODO: what do we want to visualize in this case?
        return img 
    elif isinstance(head.type, SemanticSegmentation):
        if data.ndim == 4:
            data = data[0]
        if is_label:
            masks = data.bool()
        else:
            masks = seg_output_to_bool(data)
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
                if "conf_thres" not in kwargs:
                    kwargs["conf_thres"] = 0.3
                if "iou_thres" not in kwargs:
                    kwargs["iou_thres"] = 0.6

                output_nms = yolov6_out2box(data, head, **kwargs)[0]
                bboxs = output_nms[:,:4]
                labels = output_nms[:,5].int()

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
            return img
        
def torch_to_cv2(img, to_rgb=False):
    if img.is_floating_point():
        img = img.mul(255).int()
    img = torch.clamp(img, 0, 255)
    img = np.transpose(img.cpu().numpy().astype(np.uint8), (1, 2, 0))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def unnormalize(img, original_mean=(0.485, 0.456, 0.406), 
        original_std=(0.229, 0.224, 0.225), to_uint8=False):
    mean = np.array(original_mean)
    std = np.array(original_std)
    new_mean = -mean/std
    new_std = 1/std
    out_img = F.normalize(img, mean=new_mean,std=new_std)
    if to_uint8:
        out_img = torch.clamp(out_img.mul(255), 0, 255).to(torch.uint8)
    return out_img

def seg_output_to_bool(data):
    masks = torch.empty_like(data, dtype=torch.bool)
    if data.shape[0] == 1:
        classes = torch.sigmoid(data)
        masks[0] = classes > 0.4
    else:
        classes = torch.argmax(data, dim=0)
        for i in range(masks.shape[0]):
            masks[i] = classes == i
    return masks
