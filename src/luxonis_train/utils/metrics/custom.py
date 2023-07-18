import torch
from torch import Tensor
from typing import List, Optional
from torchmetrics import Metric


class ObjectKeypointSimilarity(Metric):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = True
    full_state_update: bool = True
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 1.0

    pred_keypoints: List[Tensor]
    groundtruth_keypoints: List[Tensor]
    groundtruth_scales: List[Tensor]

    def __init__(
        self, num_keypoints: int, kpt_sigmas: Optional[torch.Tensor] = None, **kwargs
    ) -> None:
        """Object Keypoint Similarity metric for evaluating keypoint predictions

        Args:
            num_keypoints (int): Number of keypoints
            kpt_sigmas (Optional[torch.Tensor], optional): Sigma for each keypoint to weigh its importance,
                if None use same weights for all. Defaults to None.

        As input to ``forward`` and ``update`` the metric accepts the following input:
        - preds (list): A list consisting of dictionaries each containg key-values for a single image.
            Parameters that should be provided per dict:
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.

        - `target` (list): A list consisting of dictionaries each containg key-values for a single image.
            Parameters that should be provided per dict:
            - keypoints (torch.FloatTensor): Tensor of shape (N, 3*K) and in format [x,y,vis,x,y,vis,...] where `x` an `y`
                are unnormalized keypoint coordinates and `vis` is keypoint visibility.
            - scales (torch.FloatTensor): Tensor of shape (N) where each value corresponds to scale of the bounding box.
                Scale of one bounding box is defined as sqrt(width*height) where width and height are unnormalized.

        """
        super().__init__(**kwargs)

        self.num_keypoints = num_keypoints
        self.kpt_sigmas = kpt_sigmas or torch.ones(num_keypoints)

        self.add_state("pred_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_keypoints", default=[], dist_reduce_fx=None)
        self.add_state("groundtruth_scales", default=[], dist_reduce_fx=None)

    def update(self, preds: list, target: list):
        """Torchmetric update function"""
        for item in preds:
            self.pred_keypoints.append(item["keypoints"])

        for item in target:
            self.groundtruth_keypoints.append(item["keypoints"])
            self.groundtruth_scales.append(item["scales"])

    def compute(self):
        """Torchmetric compute function"""
        images_oks = torch.zeros(len(self.groundtruth_keypoints))
        # iterate over all images
        for i, (pred_kpts, gt_kpts, gt_scales) in enumerate(
            zip(
                self.pred_keypoints, self.groundtruth_keypoints, self.groundtruth_scales
            )
        ):
            curr_oks = torch.zeros(gt_kpts.shape[0])
            # reshape tensors in [N, K, 3] format
            gt_kpts = torch.reshape(gt_kpts, (-1, self.num_keypoints, 3))
            pred_kpts = torch.reshape(pred_kpts, (-1, self.num_keypoints, 3))
            # for each image compute OKS between GT and every prediciton
            for j, curr_gt in enumerate(gt_kpts):
                curr_max = 0
                for k, curr_pred in enumerate(pred_kpts):
                    curr_max = max(
                        curr_max,
                        self._compute_oks(curr_pred, curr_gt, scale=gt_scales[j]),
                    )
                # OKS of current GT is OKS with prediction that fits the most
                curr_oks[j] = curr_max
            # OKS of current image is mean of all (gt, prediction) pairs
            images_oks[i] = curr_oks.mean()
        # Output OKS is mean over all images
        mean_oks = images_oks.mean()

        return mean_oks

    def _compute_oks(self, pred: torch.Tensor, gt: torch.Tensor, scale: float):
        """Compute Object Keypoint Similarity between ground truth and prediction
            as defined here: https://cocodataset.org/#keypoints-eval

        Args:
            pred (torch.Tensor): Prediction with shape [K, 3]
            gt (torch.Tensor): GT with shape [K, 3]
            scale (float): Scale of GT

        Returns:
            torch.FloatTensor: Object Keypoint Similarity between pred and GT
        """
        # Compute the L2/Euclidean Distance
        distances = torch.norm(pred[:, :2] - gt[:, :2], dim=-1)
        # Compute the exponential part of the equation
        exp_vector = torch.exp(
            -(distances**2) / (2 * (scale**2) * (self.kpt_sigmas**2))
        )
        numerator = torch.dot(exp_vector, gt[:, 2].bool().float())
        denominator = torch.sum(gt[:, 2].bool().int()) + 1e-9
        return numerator / denominator
