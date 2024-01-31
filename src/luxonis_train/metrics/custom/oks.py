from typing import Dict, List, Optional

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.metrics.luxonis_metric import LuxonisMetric
from luxonis_train.utils.boxutils import non_max_suppression
from luxonis_train.utils.types import Labels, LabelType, ModulePacket


class ObjectKeypointSimilarity(LuxonisMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_keypoints: List[Tensor]
    groundtruth_keypoints: List[Tensor]
    groundtruth_scales: List[Tensor]

    # TODO: better name handling
    name = "OKS"

    def __init__(
        self,
        num_keypoints: int,
        n_classes: int,
        original_in_shape,
        kpt_sigmas: Optional[Tensor] = None,
        use_cocoeval_oks: bool = False,
        **kwargs,
    ) -> None:
        """Object Keypoint Similarity metric for evaluating keypoint predictions

        Args:
            num_keypoints (int): Number of keypoints
            kpt_sigmas (Optional[Tensor], optional): Sigma for each keypoint to
            weigh its importance,
                if None use same weights for all. Defaults to None.
            use_cocoeval_oks (bool, optional): Whether to use same OKS formula as
            in COCOeval or use
                the one from definition. Defaults to False.

        As input to ``forward`` and ``update`` the metric accepts the following input:
        - preds (List[Dict[str, Tensor]]): A list consisting of dictionaries
        each containing key-values for a single image.
            Parameters that should be provided per dict:
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format
            [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.

        - `target` (List[Dict[str, Tensor]]): A list consisting of
        dictionaries each containing key-values for a single image.
            Parameters that should be provided per dict:
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format
            [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.
            - scales (torch.FloatTensor): Tensor of shape (N) where each value
            corresponds to scale of the bounding box.
                Scale of one bounding box is defined as sqrt(width*height) where
                width and height are unnormalized.

        """
        super().__init__(**kwargs)

        self.n_classes = n_classes
        self.num_keypoints = num_keypoints
        self.original_in_shape = original_in_shape
        if kpt_sigmas is not None and len(kpt_sigmas) != num_keypoints:
            raise ValueError("Expected kpt_sigmas to be of shape (num_keypoints).")
        self.kpt_sigmas = kpt_sigmas or torch.ones(num_keypoints)
        self.use_cocoeval_oks = use_cocoeval_oks

        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_scales", default=[], dist_reduce_fx=None)

    def preprocess(self, outputs: ModulePacket, labels: Labels):
        kpts = labels[LabelType.KEYPOINT]
        boxes = labels[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        label = torch.zeros((len(boxes), nkpts * 3 + 6))
        label[:, :2] = boxes[:, :2]
        label[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "xyxy")
        label[:, 6::3] = kpts[:, 2::3]  # insert kp x coordinates
        label[:, 7::3] = kpts[:, 3::3]  # insert kp y coordinates
        label[:, 8::3] = kpts[:, 4::3]  # insert kp visibility

        nms = non_max_suppression(
            outputs["keypoints"][0],
            n_classes=self.n_classes,
            conf_thres=0.03,
            iou_thres=0.3,
            box_format="cxcywh",
        )
        output_list_oks = []
        label_list_oks = []
        image_size = self.original_in_shape[2:]

        for i in range(len(nms)):
            output_list_oks.append({"keypoints": nms[i][:, 6:]})

            curr_label = label[label[:, 0] == i].to(nms[i].device)
            curr_bboxs = curr_label[:, 2:6]
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            curr_kpts = curr_label[:, 6:]
            curr_kpts[:, 0::3] *= image_size[1]
            curr_kpts[:, 1::3] *= image_size[0]
            curr_bboxs_widths = curr_bboxs[:, 2] - curr_bboxs[:, 0]
            curr_bboxs_heights = curr_bboxs[:, 3] - curr_bboxs[:, 1]
            curr_scales = torch.sqrt(curr_bboxs_widths * curr_bboxs_heights)
            label_list_oks.append({"keypoints": curr_kpts, "scales": curr_scales})

        return output_list_oks, label_list_oks

    def update(
        self, preds: List[Dict[str, Tensor]], target: List[Dict[str, Tensor]]
    ) -> None:
        """Torchmetric update function"""
        for item in preds:
            keypoints = fix_empty_tensors(item["keypoints"])
            self.pred_keypoints.append(keypoints)

        for item in target:
            keypoints = fix_empty_tensors(item["keypoints"])
            self.groundtruth_keypoints.append(keypoints)
            self.groundtruth_scales.append(item["scales"])

    def compute(self) -> Tensor:
        """Torchmetric compute function"""
        self.kpt_sigmas = self.kpt_sigmas.to(
            self.device
        )  # explicitly move to current device
        image_mean_oks = torch.zeros(len(self.groundtruth_keypoints))
        # iterate over all images
        for i, (pred_kpts, gt_kpts, gt_scales) in enumerate(
            zip(
                self.pred_keypoints, self.groundtruth_keypoints, self.groundtruth_scales
            )
        ):
            gt_kpts = torch.reshape(gt_kpts, (-1, self.num_keypoints, 3))  # [N, K, 3]
            pred_kpts = torch.reshape(
                pred_kpts, (-1, self.num_keypoints, 3)
            )  # [M, K, 3]

            # compute OKS
            image_ious = self._compute_oks(pred_kpts, gt_kpts, gt_scales)  # [M, N]

            # perform linear sum assignment
            gt_indices, pred_indices = linear_sum_assignment(
                image_ious.cpu().numpy(), maximize=True
            )
            matched_ious = [image_ious[n, m] for n, m in zip(gt_indices, pred_indices)]
            # take mean as OKS for image
            image_mean_oks[i] = torch.tensor(matched_ious).mean()

        # final prediction is mean of OKS over all images
        final_oks = image_mean_oks.nanmean()

        return final_oks

    def _compute_oks(self, pred: Tensor, gt: Tensor, scales: Tensor) -> Tensor:
        """Compute Object Keypoint Similarity between every GT and prediction

        Args:
            pred (Tensor): Prediction tensor with shape [N, K, 3]
            gt (Tensor): GT tensor with shape [M, K, 3]
            scale (float): Scale tensor for every GT [M,]

        Returns:
            Tensor: Object Keypoint Similarity every pred and gt [M, N]
        """
        eps = 1e-7
        distances = (gt[:, None, :, 0] - pred[..., 0]) ** 2 + (
            gt[:, None, :, 1] - pred[..., 1]
        ) ** 2  # (N, M, 17)
        kpt_mask = gt[..., 2] != 0  # only compute on visible keypoints
        if self.use_cocoeval_oks:
            # use same formula as in COCOEval script here:
            # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229
            oks = (
                distances
                / (2 * self.kpt_sigmas) ** 2
                / (scales[:, None, None] + eps)
                / 2
            )
        else:
            # use same formula as defined here: https://cocodataset.org/#keypoints-eval
            oks = (
                distances
                / ((scales[:, None, None] + eps) * self.kpt_sigmas.to(scales.device))
                ** 2
                / 2
            )

        return (torch.exp(-oks) * kpt_mask[:, None]).sum(-1) / (
            kpt_mask.sum(-1)[:, None] + eps
        )


def fix_empty_tensors(input_tensor: Tensor) -> Tensor:
    """Empty tensors can cause problems in DDP mode, this methods corrects them."""
    if input_tensor.numel() == 0 and input_tensor.ndim == 1:
        return input_tensor.unsqueeze(0)
    return input_tensor
