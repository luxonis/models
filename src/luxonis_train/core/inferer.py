from typing import Literal

import cv2

from luxonis_train.attached_modules.visualizers import (
    get_unnormalized_images,
)

from .trainer import Trainer


class Inferer(Trainer):
    def __init__(
        self,
        cfg: str | dict,
        opts: list[str] | tuple[str, ...] | None,
        view: Literal["train", "test", "val"],
    ):
        opts = list(opts or [])
        opts += ["train.batch_size", "1"]
        super().__init__(cfg, opts)
        if view == "train":
            self.loader = self.pytorch_loader_train
        elif view == "test":
            self.loader = self.pytorch_loader_test
        else:
            self.loader = self.pytorch_loader_val

    def infer(self):
        self.lightning_module.eval()
        for img, labels in self.loader:
            unnorm_img = get_unnormalized_images(self.cfg, img)[0]
            outputs = self.lightning_module.forward(
                img, labels, image=unnorm_img, compute_visualizations=True
            )

            for node_name, visualizations in outputs.visualizations.items():
                for viz_name, viz in visualizations.items():
                    viz_arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                    viz_arr = cv2.cvtColor(viz_arr, cv2.COLOR_RGB2BGR)
                    name = f"{node_name}/{viz_name}"
                    cv2.imshow(name, viz_arr)
            if cv2.waitKey(0) == ord("q"):
                exit()
