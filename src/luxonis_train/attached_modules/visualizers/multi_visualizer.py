from torch import Tensor

from luxonis_train.utils.registry import VISUALIZERS
from luxonis_train.utils.types import (
    Labels,
    Packet,
)

from .luxonis_visualizer import LuxonisVisualizer


class MultiVisualizer(LuxonisVisualizer[Packet[Tensor], Labels]):
    def __init__(self, visualizers: list[dict], **kwargs):
        super().__init__(**kwargs)
        self.visualizers = []
        for item in visualizers:
            visualizer_params = item.get("params", {})
            visualizer = VISUALIZERS.get(item["name"])(**visualizer_params, **kwargs)
            self.visualizers.append(visualizer)

    def prepare(
        self, output: Packet[Tensor], label: Labels, idx: int = 0
    ) -> tuple[Packet[Tensor], Labels]:
        self._idx = idx
        return output, label

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        idx: int,
        outputs: Packet[Tensor],
        labels: Labels,
    ) -> tuple[Tensor, Tensor]:
        for visualizer in self.visualizers:
            match visualizer(label_canvas, prediction_canvas, outputs, labels, idx):
                case Tensor(data=prediction_viz):
                    prediction_canvas = prediction_viz
                case (Tensor(data=label_viz), Tensor(data=prediction_viz)):
                    label_canvas = label_viz
                    prediction_canvas = prediction_viz
                case _:
                    raise NotImplementedError
        return label_canvas, prediction_canvas
