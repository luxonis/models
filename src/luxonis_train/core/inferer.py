import torch
import cv2
import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from typing import Union, Optional
from tqdm import tqdm
from luxonis_ml import *

from luxonis_train.utils.config import Config
from luxonis_train.utils.augmentations import TrainAugmentations, ValAugmentations, Augmentations
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *
from luxonis_train.utils.general import *
from luxonis_train.utils.visualization import *


class Inferer(pl.LightningModule):
    def __init__(self, cfg: Union[str, dict], args: Optional[dict] = None):
        """Main API which is used for inference/visualization on the dataset

        Args:
            cfg (Union[str, dict]): path to config file or config dict used to setup training
            args (Optional[dict]): argument dict provided through command line, used for config overriding
        """
        super().__init__()

        self.cfg = Config(cfg)
        if args and args["override"]:
            self.cfg.override_config(args["override"])

        self.model = Model()
        self.model.build_model()

        self.load_checkpoint(self.cfg.get("model.pretrained"))
        self.model.eval()

        self.augmentations = None

    def load_checkpoint(self, path: str):
        """ Loads checkpoint weights from provided path """
        print(f"Loading weights from: {path}")
        state_dict = torch.load(path)["state_dict"]
        # remove weights that are not part of the model
        removed = []
        for key in state_dict.keys():
            if not key.startswith("model"):
                removed.append(key)
                state_dict.pop(key)
        if len(removed):
            print(f"Following weights weren't loaded: {removed}")

        self.load_state_dict(state_dict)

    def override_augmentations(self, aug: object):
        """ Overrides augmentations used for validation dataset """
        self.augmentations = aug
    
    def forward(self, inputs: torch.Tensor):
        """ Forward function used in inference """
        outputs = self.model(inputs)
        return outputs

    def infer(self):
        """ Runs inference on all images in the dataset """

        with LuxonisDataset(
            team_name=self.cfg.get("dataset.team_id"),
            dataset_name=self.cfg.get("dataset.dataset_id"),
            bucket_type=self.cfg.get("dataset.bucket_type"),
            override_bucket_type=self.cfg.get("dataset.override_bucket_type")
        ) as dataset:

            view = self.cfg.get("inferer.dataset_view")
            
            if self.augmentations == None:
                if view == "train":
                    self.augmentations = TrainAugmentations()
                else:
                    self.augmentations = ValAugmentations()

            loader_val = LuxonisLoader(
                dataset,
                view=view,
                augmentations=self.augmentations
            )

            pytorch_loader_val = torch.utils.data.DataLoader(
                loader_val,
                batch_size=self.cfg.get("train.batch_size"),
                num_workers=self.cfg.get("train.num_workers"),
                collate_fn=loader_val.collate_fn
            )

            display = self.cfg.get("inferer.display")
            save_dir = self.cfg.get("inferer.infer_save_directory")

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)

            counter = 0
            with torch.no_grad():
                for data in tqdm(pytorch_loader_val):
                    inputs = data[0]
                    labels = data[1]
                    outputs = self.forward(inputs)

                    for i, output in enumerate(outputs):
                        curr_head = self.model.heads[i]
                        curr_head_name = get_head_name(curr_head, i)
                        curr_label = get_current_label(curr_head.type, labels)

                        label_imgs = draw_on_images(inputs, curr_label, curr_head, is_label=True)
                        output_imgs = draw_on_images(inputs, output, curr_head, is_label=False)  
                        merged_imgs = [cv2.hconcat([l_img, o_img]) for l_img, o_img in zip(label_imgs, output_imgs)]
                        
                        for img in merged_imgs:
                            counter += 1
                            plt.imshow(img)
                            plt.title(curr_head_name+f"\n(labels left, outputs right)")
                            if save_dir is not None:
                                plt.savefig(os.path.join(save_dir, f"{counter}.png"))
                            if display:
                                plt.show()


    def infer_image(self, img: np.ndarray, augmentations: Optional[Augmentations] = None, 
        display: bool = True, save_path: Optional[str] = None):
        """ Runs inference on single image

        Args:
            img (np.ndarray): Input image of shape (H x W x C) and dtype uint8.
            augmentations (Optional[Augmentations], optional): Instance of augmentation class. If None use ValAugmentations(). Defaults to None.
            display (bool, optional): Control if want to display output. Defaults to True.
            save_path (Optional[str], optional): Path for saving the output, will generate separate image for each model head. If None then don't save. Defaults to None.
        """

        if augmentations == None:
            augmentations = ValAugmentations()

        # IMG IN RGB HWC
        transformed = augmentations.transform(
            image = img,
            bboxes = [],
            bbox_classes = [],
            keypoints = [],
            keypoints_classes = []
        )
        inputs = torch.unsqueeze(transformed["image"], dim=0)
        outputs = self.forward(inputs)

        for i, output in enumerate(outputs):
            curr_head = self.model.heads[i]
            curr_head_name = get_head_name(curr_head, i)

            output_img = draw_on_images(inputs, output, curr_head, is_label=False)[0]  
            
            if save_path is not None:
                path, save_type = save_path.rsplit(".", 1) # get desired save type (e.g. .png, .jpg, ...)
                # use cv2 for saving to avoid paddings
                save_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(path+f"_{curr_head_name}.{save_type}", save_img)

            if display:
                plt.imshow(output_img)
                plt.title(curr_head_name)
                plt.show()
