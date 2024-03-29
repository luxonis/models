import argparse
import torch
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.loader import LuxonisLoader, TrainAugmentations, ValAugmentations

from luxonis_train.utils.config import Config
from luxonis_train.utils.visualization import draw_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_argument("--override", default=None, type=str, help="Manually override config parameter")
    parser.add_argument("--view", type=str, default="val", help="Dataset view to use")
    parser.add_argument("--no-display", action="store_true", help="Don't display images")
    parser.add_argument("--save-dir", type=str, default=None, help="Path to save directory, by default don't save")
    args = parser.parse_args()
    
    cfg = Config(args.config)
    if args.override:
        cfg.override_config(args.override)

    load_dotenv()

    image_size = cfg.get("train.preprocessing.train_image_size")
    
    with LuxonisDataset(
        team_id=cfg.get("dataset.team_id"),
        dataset_id=cfg.get("dataset.dataset_id"),
        bucket_type=cfg.get("dataset.bucket_type"),
        override_bucket_type=cfg.get("dataset.override_bucket_type")
    ) as dataset:
    
        augmentations = TrainAugmentations(
            image_size=image_size,
            augmentations=cfg.get("train.preprocessing.augmentations"),
            train_rgb=cfg.get("train.preprocessing.train_rgb"),
            keep_aspect_ratio=cfg.get("train.preprocessing.keep_aspect_ratio")
        ) if args.view == "train" else ValAugmentations(
            image_size=image_size,
            augmentations=cfg.get("train.preprocessing.augmentations"),
            train_rgb=cfg.get("train.preprocessing.train_rgb"),
            keep_aspect_ratio=cfg.get("train.preprocessing.keep_aspect_ratio")
        )

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

        save_dir = args.save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        counter = 0
        for data in pytorch_loader_train:
            imgs, label_dict = data
            out_imgs = draw_labels(
                imgs = imgs, label_dict = label_dict, 
                unnormalize_img = cfg.get("train.preprocessing.normalize.active"),
                cvt_color = not cfg.get("train.preprocessing.train_rgb")
            )

            for i in out_imgs:
                plt.imshow(i)
                if save_dir is not None:
                    counter += 1
                    plt.savefig(os.path.join(save_dir, f"{counter}.png"))
                if not args.no_display:
                    plt.show()