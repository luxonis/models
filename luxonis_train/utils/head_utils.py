import torch
from luxonis_train.utils.assigners.anchor_generator import generate_anchors
from luxonis_train.utils.boxutils import dist2bbox, non_max_suppression

# utils specific to some head (preprocessing or postprocessing functions)

def yolov6_out2box(output, head, **kwargs):
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