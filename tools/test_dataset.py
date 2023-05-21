import argparse
import torch
import yaml

from luxonis_ml import *
from luxonis_train.utils.config import Config
from luxonis_train.utils.augmentations import TrainAugmentations
from luxonis_train.utils.visualization import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_arguemnt("-v", "--view", type=str, defualt="val", help="Dataset view to use")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    cfg = Config(cfg)
    image_size = cfg.get("train.preprocessing.train_image_size")

    with LuxonisDataset(
        team_name=cfg.get("dataset.team_name"),
        dataset_name=cfg.get("dataset.dataset_name")
    ) as dataset:

        classes, classes_by_task = dataset.get_classes()
        colors = [
            (np.random.randint(256),np.random.randint(256),np.random.randint(256)) \
            for _ in range(len(classes))
        ]

        train_augmentations = TrainAugmentations()

        loader_train = LuxonisLoader(
            dataset,
            view='val',
            augmentations=train_augmentations
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
            for i in range(len(imgs)): # supports batch_size>1
                img = unnormalize(imgs[i], to_uint8=True)

                for label_type in anno_dict:
                    if label_type == "keypoints":
                        curr_label = anno_dict[label_type]
                        curr_label = curr_label[curr_label[:,0]==i][:,1:]

                        keypoints_flat = torch.reshape(curr_label[:,1:], (-1,3))
                        keypoints_points = keypoints_flat[:,:2]
                        keypoints_points[:,0] *= iw
                        keypoints_points[:,1] *= ih
                        keypoints_visibility = keypoints_flat[:,2]

                        # torchvision expects format [num_instances, K, 2]
                        out_keypoints = torch.reshape(keypoints_points, (-1, 17, 2)).int()
                        img = draw_keypoints(img, out_keypoints, colors="red")
                    if label_type == "class":
                        curr_label = anno_dict[label_type][i]
                        print(f"Class: {curr_label}")
                    if label_type == "segmentation":
                        curr_label = anno_dict[label_type][i]
                        masks = curr_label.bool()
                        img = draw_segmentation_masks(img, masks, alpha=0.4, colors=colors)
                    if label_type == "bbox":
                        curr_label = anno_dict[label_type]
                        curr_label = curr_label[curr_label[:,0]==i]
                        bboxs = xywh2xyxy_coco(curr_label[:, 2:])
                        bboxs[:, 0::2] *= iw
                        bboxs[:, 1::2] *= ih
                        img = draw_bounding_boxes(img, bboxs)

                img_output = torch_img_to_numpy(img)
                plt.imshow(img_output)
                plt.title(f"{i}")
                plt.show()
