import torch
from torchmetrics import Metric
import warnings


class ObjectKeypointSimilarity(Metric):
    def __init__(self):
        super().__init__()
        # self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds, target = self._input_format(preds, target)
        # assert preds.shape == target.shape

        # self.correct += torch.sum(preds == target)
        # self.total += target.numel()
        pass

    def compute(self):
        # return self.correct.float() / self.total
        warnings.warn(
            "ObjectKeypointSimilarity metric not yet implemented. Returning default value 1."
        )
        return torch.ones(1)
