import torch
import cv2
import os
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from typing import Union, Optional
from dotenv import load_dotenv
from tqdm import tqdm
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.loader import LuxonisLoader
from luxonis_ml.loader import TrainAugmentations, ValAugmentations, Augmentations

from luxonis_train.utils.config import ConfigHandler
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.visualization import draw_outputs, draw_labels
from luxonis_train.utils.filesystem import LuxonisFileSystem


class Inferer(pl.LightningModule):
    def __init__(self, cfg: Union[str, dict], args: Optional[dict] = None):
        """Main API which is used for inference/visualization on the dataset

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (Optional[dict]): argument dict provided through command line, used for config overriding
        """
        super().__init__()

        self.cfg = ConfigHandler(cfg)
        if args and args["override"]:
            self.cfg.override_config(args["override"])
        load_dotenv()

        self.model = Model()
        self.model.build_model()

        self.load_checkpoint(self.cfg.get("model.pretrained"))
        self.model.eval()

        self.augmentations = None

    def load_checkpoint(self, path: str):
        """Loads checkpoint weights from provided path"""
        print(f"Loading weights from: {path}")
        fs = LuxonisFileSystem(path)
        checkpoint = torch.load(fs.read_to_byte_buffer())
        state_dict = checkpoint["state_dict"]
        # remove weights that are not part of the model
        removed = []
        for key in list(state_dict.keys()):
            if not key.startswith("model"):
                removed.append(key)
                state_dict.pop(key)
        if len(removed):
            print(f"Following weights weren't loaded: {removed}")

        self.load_state_dict(state_dict)

    def override_augmentations(self, aug: object):
        """Overrides augmentations used for validation dataset"""
        self.augmentations = aug

    def forward(self, inputs: torch.Tensor):
        """Forward function used in inference"""
        outputs = self.model(inputs)
        return outputs

    def infer(self):
        """Runs inference on all images in the dataset"""

        dataset = LuxonisDataset(
            dataset_name=self.cfg.get("dataset.dataset_name"),
            team_id=self.cfg.get("dataset.team_id"),
            dataset_id=self.cfg.get("dataset.dataset_id"),
            bucket_type=self.cfg.get("dataset.bucket_type"),
            bucket_storage=self.cfg.get("dataset.bucket_storage"),
        )
        view = self.cfg.get("inferer.dataset_view")

        if self.augmentations == None:
            if view == "train":
                self.augmentations = TrainAugmentations(
                    image_size=self.cfg.get("train.preprocessing.train_image_size"),
                    augmentations=[
                        i.model_dump()
                        for i in self.cfg.get("train.preprocessing.augmentations")
                    ],
                    train_rgb=self.cfg.get("train.preprocessing.train_rgb"),
                    keep_aspect_ratio=self.cfg.get(
                        "train.preprocessing.keep_aspect_ratio"
                    ),
                )
            else:
                self.augmentations = ValAugmentations(
                    image_size=self.cfg.get("train.preprocessing.train_image_size"),
                    augmentations=[
                        i.model_dump()
                        for i in self.cfg.get("train.preprocessing.augmentations")
                    ],
                    train_rgb=self.cfg.get("train.preprocessing.train_rgb"),
                    keep_aspect_ratio=self.cfg.get(
                        "train.preprocessing.keep_aspect_ratio"
                    ),
                )

        loader_val = LuxonisLoader(
            dataset,
            view=view,
            augmentations=self.augmentations,
        )

        pytorch_loader_val = torch.utils.data.DataLoader(
            loader_val,
            batch_size=self.cfg.get("train.batch_size"),
            num_workers=self.cfg.get("train.num_workers"),
            collate_fn=loader_val.collate_fn,
        )

        display = self.cfg.get("inferer.display")
        save_dir = self.cfg.get("inferer.infer_save_directory")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        unnormalize_img = self.cfg.get("train.preprocessing.normalize.active")
        normalize_params = self.cfg.get("train.preprocessing.normalize.params")
        cvt_color = not self.cfg.get("train.preprocessing.train_rgb")
        counter = 0
        with torch.no_grad():
            for data in tqdm(pytorch_loader_val):
                inputs = data[0]
                label_dict = data[1]
                outputs = self.forward(inputs)

                for i, output in enumerate(outputs):
                    curr_head = self.model.heads[i]
                    curr_head_name = curr_head.get_name(i)

                    label_imgs = draw_labels(
                        imgs=inputs,
                        label_dict=label_dict,
                        label_keys=curr_head.label_types,
                        unnormalize_img=unnormalize_img,
                        cvt_color=cvt_color,
                        overlay=True,
                        normalize_params=normalize_params,
                    )
                    output_imgs = draw_outputs(
                        imgs=inputs,
                        output=output,
                        head=curr_head,
                        unnormalize_img=unnormalize_img,
                        cvt_color=cvt_color,
                        normalize_params=normalize_params,
                    )
                    merged_imgs = [
                        cv2.hconcat([l_img, o_img])
                        for l_img, o_img in zip(label_imgs, output_imgs)
                    ]

                    for img in merged_imgs:
                        counter += 1
                        plt.imshow(img)
                        plt.title(curr_head_name + f"\n(labels left, outputs right)")
                        if save_dir is not None:
                            plt.savefig(os.path.join(save_dir, f"{counter}.png"))
                        if display:
                            plt.show()

    def infer_image(
        self,
        img: np.ndarray,
        augmentations: Optional[Augmentations] = None,
        display: bool = True,
        save_path: Optional[str] = None,
    ):
        """Runs inference on single image

        Args:
            img (np.ndarray): Input image of shape (H x W x C) and dtype uint8.
            augmentations (Optional[Augmentations], optional): Instance of augmentation class. If None use ValAugmentations(). Defaults to None.
            display (bool, optional): Control if want to display output. Defaults to True.
            save_path (Optional[str], optional): Path for saving the output, will generate separate image for each model head. If None then don't save. Defaults to None.
        """

        if augmentations == None:
            augmentations = ValAugmentations(
                image_size=self.cfg.get("train.preprocessing.train_image_size"),
                augmentations=[
                    i.model_dump()
                    for i in self.cfg.get("train.preprocessing.augmentations")
                ],
                train_rgb=self.cfg.get("train.preprocessing.train_rgb"),
                keep_aspect_ratio=self.cfg.get("train.preprocessing.keep_aspect_ratio"),
            )

        # IMG IN RGB HWC
        transformed = augmentations.transform(
            image=img, bboxes=[], bbox_classes=[], keypoints=[], keypoints_classes=[]
        )
        inputs = torch.unsqueeze(transformed["image"], dim=0)
        outputs = self.forward(inputs)

        unnormalize_img = self.cfg.get("train.preprocessing.normalize.active")
        cvt_color = not self.cfg.get("train.preprocessing.train_rgb")

        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = curr_head.get_name(i)

            output_img = draw_outputs(
                imgs=inputs,
                output=output,
                head=curr_head,
                unnormalize_img=unnormalize_img,
                cvt_color=cvt_color,
            )[0]

            if save_path is not None:
                path, save_type = save_path.rsplit(
                    ".", 1
                )  # get desired save type (e.g. .png, .jpg, ...)
                # use cv2 for saving to avoid paddings
                save_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path + f"_{curr_head_name}.{save_type}", save_img)

            if display:
                plt.imshow(output_img)
                plt.title(curr_head_name)
                plt.show()
