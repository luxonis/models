import torch
import cv2
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from luxonis_ml import *

from luxonis_train.utils.augmentations import ValAugmentations
from luxonis_train.utils.config import *
from luxonis_train.models import Model
from luxonis_train.models.heads import *
from luxonis_train.utils.head_type import *
from luxonis_train.utils.general import *
from luxonis_train.utils.visualization import *


class Inferer(pl.LightningModule):
    def __init__(self, cfg: dict):
        """Main API which is used for inference/visualization on the dataset

        Args:
            cfg (dict): Configs dict used to setup inference.
        """
        super().__init__()
        self.cfg = cfg

        # check if model is predefined
        if self.cfg["model"]["type"]:
            self.load_predefined_cfg(self.cfg["model"]["type"])
        
        self.model = Model()
        self.model.build_model(self.cfg["model"], self.cfg["train"]["image_size"])

        self.load_checkpoint(self.cfg["model"]["pretrained"])
        self.model.eval()
        
        self.val_augmentations = None

    def load_checkpoint(self, path):
        """ Load checkpoint weights from provided path """
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

    def forward(self, inputs):
        """ Forward function used in inference """
        outputs = self.model(inputs)
        return outputs
    
    def infer(self, display:bool = True, save:bool = False):
        """Runs inference on all images in the dataset

        Args:
            display (bool, optional): Display output for each head. Defaults to True.
            save (bool, optional): Save output image. Defaults to False.
        """
        with LuxonisDataset(
            local_path=self.cfg["dataset"]["local_path"] if "local_path" in self.cfg["dataset"] else None,
            s3_path=self.cfg["dataset"]["s3_path"] if "s3_path" in self.cfg["dataset"] else None
        ) as dataset:
            
            if self.val_augmentations == None:
                self.val_augmentations = ValAugmentations(image_size=self.cfg["train"]["image_size"])

            loader_val = LuxonisLoader(dataset, view="val")
            loader_val.map(loader_val.auto_preprocess)
            loader_val.map(self.val_augmentations)
            pytorch_loader_val = loader_val.to_pytorch(
                batch_size=1,
                num_workers=self.cfg["train"]["n_workers"]
            )

            for data in pytorch_loader_val:
                inputs = data[0].float()
                img = unnormalize(inputs[0], to_uint8=True)
                labels = data[1:]
                outputs = self.forward(inputs)

                for i, output in enumerate(outputs):
                    curr_head = self.model.heads[i]
                    curr_head_name = get_head_name(curr_head, i)
                    curr_label = get_current_label(curr_head.type, labels)
                    
                    img_labels = draw_on_image(img, curr_label, curr_head, is_label=True)
                    img_labels = torch_to_cv2(img_labels, to_rgb=True)
                    img_outputs = draw_on_image(img, output, curr_head)
                    img_outputs = torch_to_cv2(img_outputs, to_rgb=True)
                    
                    out_img = cv2.hconcat([img_labels, img_outputs])
                    if display:
                        plt.imshow(out_img)
                        plt.title(curr_head_name+f"\n(labels left, outputs right)")
                        plt.show()
                        # TODO: this freezes for loop, needs to be fixed and then set to_rgb=False 
                        # cv2.imshow(curr_head_name, out_img)
                        # key = cv2.waitKey(0)
                    if save:
                        # TODO: What do we want to save?
                        pass

    def infer_image(self, img:torch.tensor, display:bool = True, save:bool = False):
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
