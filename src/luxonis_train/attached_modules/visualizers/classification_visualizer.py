# TODO: docs
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from .base_visualizer import BaseVisualizer
from .utils import (
    figure_to_torch,
    numpy_to_torch_img,
    torch_img_to_numpy,
)


class ClassificationVisualizer(BaseVisualizer[Tensor, Tensor]):
    def __init__(
        self, class_names: list[str] | None = None, include_plot: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.class_names = class_names
        self.include_plot = include_plot

    def _get_class_name(self, cls: Tensor) -> str:
        if self.class_names is None:
            return str(int(cls.item()))
        return self.class_names[int(cls.item())]

    def _generate_plot(self, prediction: Tensor, width: int, height: int) -> Tensor:
        prediction = prediction.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.bar(np.arange(len(prediction)), prediction)
        ax.set_xticks(np.arange(len(prediction)))
        ax.set_xticklabels(np.arange(1, len(prediction) + 1))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.grid(True)
        return figure_to_torch(fig, width, height)

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        idx: int,
        prediction: Tensor,
        label: Tensor,
    ) -> Tensor | tuple[Tensor, Tensor]:
        prediction = prediction[idx]
        gt = self._get_class_name(label[idx])
        arr = torch_img_to_numpy(label_canvas)
        curr_class = self._get_class_name(prediction.argmax() + 1)
        arr = cv2.putText(
            arr,
            f"GT: {gt}, predicted: {curr_class}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        overlay = numpy_to_torch_img(arr)
        if self.include_plot:
            plot = self._generate_plot(
                prediction, prediction_canvas.shape[2], prediction_canvas.shape[1]
            )
            return overlay, plot
        return overlay
