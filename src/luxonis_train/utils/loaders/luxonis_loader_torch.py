import torch
import numpy as np
from typing import Optional
from luxonis_ml.data import LuxonisLoader

from .base_loader import BaseLoaderTorch, LuxonisLoaderTorchOutput
from luxonis_train.utils.registry import LOADERS


@LOADERS.register_module()
class LuxonisLoaderTorch(BaseLoaderTorch):
    def __init__(
        self,
        dataset: "luxonis_ml.data.LuxonisDataset",
        view: str = "train",
        stream: bool = False,
        augmentations: Optional["luxonis_ml.data.Augmentations"] = None,
        mode: str = "fiftyone",
    ):
        self.base_loader = LuxonisLoader(
            dataset=dataset,
            view=view,
            stream=stream,
            augmentations=augmentations,
            mode=mode,
        )

    def __len__(self) -> int:
        return len(self.base_loader)

    def __getitem__(self, idx: int) -> LuxonisLoaderTorchOutput:
        img, annotations = self.base_loader[idx]

        # convert img and annotations to torch tensors
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.tensor(img)
        for key in annotations:
            annotations[key] = torch.tensor(annotations[key])

        return img, annotations
