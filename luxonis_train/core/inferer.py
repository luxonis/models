import torch
import cv2
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from typing import Union
from luxonis_ml import *

from luxonis_train.utils.config import Config
from luxonis_train.utils.augmentations import ValAugmentations
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *
from luxonis_train.utils.general import *
from luxonis_train.utils.visualization import *


class Inferer(pl.LightningModule):
    def __init__(self, args: dict, cfg: Union[str, dict]):
        """Main API which is used for inference/visualization on the dataset

        Args:
            args (dict): argument dict provided through command line, used for config overriding
            cfg (Union[str, dict]): path to config file or config dict used to setup training
        """
        super().__init__()

        self.cfg = Config(cfg)
        if args["override"]:
            self.cfg.override_config(args["override"])
        
        self.model = Model()
        self.model.build_model()

        self.load_checkpoint(self.cfg.get("model.pretrained"))
        self.model.eval()
        
        self.val_augmentations = None

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

    def forward(self, inputs: torch.Tensor):
        """ Forward function used in inference """
        outputs = self.model(inputs)
        return outputs
    
    def infer(self, display: bool = True, save: bool = False):
        """Runs inference on all images in the dataset

        Args:
            display (bool, optional): Display output for each head. Defaults to True.
            save (bool, optional): Save output image. Defaults to False.
        """
        with LuxonisDataset(
            local_path=self._data["dataset"]["local_path"],
            s3_path=self._data["dataset"]["s3_path"]
        ) as dataset:
            
            if self.val_augmentations == None:
                self.val_augmentations = ValAugmentations()

            loader_val = LuxonisLoader(dataset, 
                view=self.cfg.get("inferer.dataset_view")
            )
            loader_val.map(loader_val.auto_preprocess)
            loader_val.map(self.val_augmentations)
            # TODO: Do we want this configurable?
            pytorch_loader_val = loader_val.to_pytorch(
                batch_size=1, num_workers=1
            )
            
            with torch.no_grad():
                for data in pytorch_loader_val:
                    inputs = data[0].float()
                    img = unnormalize(inputs[0], to_uint8=True)
                    labels = data[1]
                    outputs = self.forward(inputs)

                    for i, output in enumerate(outputs):
                        curr_head = self.model.heads[i]
                        curr_head_name = get_head_name(curr_head, i)
                        curr_label = get_current_label(curr_head.type, labels)

                        img_labels = draw_on_image(img, curr_label, curr_head, is_label=True)
                        img_labels = torch_img_to_numpy(img_labels)
                        img_outputs = draw_on_image(img, output, curr_head)
                        img_outputs = torch_img_to_numpy(img_outputs)
                        
                        out_img = cv2.hconcat([img_labels, img_outputs])
                        if display:
                            plt.imshow(out_img)
                            plt.title(curr_head_name+f"\n(labels left, outputs right)")
                            plt.show()
                        if save:
                            # TODO: What do we want to save?
                            pass

    def infer_image(self, img: torch.Tensor, display: bool = True, save: bool = False):
        """Runs inference on a single image

        Args:
            img (torch.tensor): Tensor of shape (C x H x W) and dtype uint8.
            display (bool, optional): Display output for each head. Defaults to True.
            save (bool, optional): Save output image. Defaults to False.

        Raises:
            NotImplementedError: Not implemented yet
        """
        # first need to run img through augmentation, then through model and after similar to infer()
        raise NotImplementedError()
