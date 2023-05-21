import argparse
import torch
import yaml

from luxonis_ml import *
from luxonis_train.utils.config import Config
from luxonis_train.utils.augmentations import ValAugmentations
from luxonis_train.utils.visualization import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_argument("-v", "--view", type=str, default="val", help="Dataset view to use")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    cfg = Config(cfg)
    image_size = cfg.get("train.preprocessing.train_image_size")

    with LuxonisDataset(
        local_path=cfg.get("dataset.local_path"),
        s3_path=cfg.get("dataset.s3_path")
    ) as dataset:
        
        train_augmentations = ValAugmentations()
        
        loader_train = LuxonisLoader(dataset, view=args.view)
        loader_train.map(loader_train.auto_preprocess)
        loader_train.map(train_augmentations)
        pytorch_loader_train = loader_train.to_pytorch(
            batch_size=4,
            num_workers=1
        )

        ih, iw = image_size[0], image_size[1]

        for data in pytorch_loader_train:
            imgs, anno_dict = data
            out_imgs = draw_only_labels(imgs, anno_dict, image_size = image_size)
            for i in out_imgs:
                plt.imshow(i)
                plt.show()