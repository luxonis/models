import argparse
import torch
from dotenv import load_dotenv

from luxonis_ml import *
from luxonis_train.utils.config import Config
from luxonis_train.utils.augmentations import TrainAugmentations, ValAugmentations
from luxonis_train.utils.visualization import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_argument("-v", "--view", type=str, default="val", help="Dataset view to use")
    parser.add_argument("--override", default=None, type=str, help="Manually override config parameter")
    args = parser.parse_args()

    cfg = Config(args.config)
    if args.override:
        cfg.override_config(args.override)

    image_size = cfg.get("train.preprocessing.train_image_size")

    with LuxonisDataset(
        team_name=cfg.get("dataset.team_name"),
        dataset_name=cfg.get("dataset.dataset_name")
    ) as dataset:
    
        augmentations = TrainAugmentations() if args.view == "train" else ValAugmentations()
        loader_train = LuxonisLoader(
            dataset,
            view=args.view,
            augmentations=augmentations
        )
        pytorch_loader_train = torch.utils.data.DataLoader(
            loader_train,
            batch_size=4,
            num_workers=1,
            collate_fn=loader_train.collate_fn
        )

        ih, iw = image_size[0], image_size[1]

        for data in pytorch_loader_train:
            imgs, anno_dict = data
            out_imgs = draw_only_labels(imgs, anno_dict, image_size = image_size)
            for i in out_imgs:
                plt.imshow(i)
                plt.show()
