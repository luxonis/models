import albumentations as A
from albumentations import *
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import torch

class TrainAugmentations:
    def __init__(self, cfg=None, image_size=[256, 256], normalize=True, to_tensor=True):
        self.cfg = cfg
        self.image_size = image_size
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.transform = self.parse_cfg()

    def parse_cfg(self):
        # Always perform Resize
        augmentations = [A.Resize(self.image_size[0], self.image_size[1])]
        if self.cfg:
            for aug in self.cfg:
                augmentations.append(
                    eval(aug["name"])(**aug["params"] if aug["params"] else {})
                )

        if self.normalize: augmentations.append(A.Normalize())
        if self.to_tensor: augmentations.append(ToTensorV2())

        return A.Compose(
            augmentations,
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
            keypoint_params=A.KeypointParams(format="xy", label_fields=["keypoints_classes"], remove_invisible=False),
        )

    def __call__(self, data):
        img, classify, bboxes, seg, keypoints = data
        img_in = img.numpy()
        # albumentations expects with RGB image with HWC format
        img_in = np.transpose(img_in, (1,2,0))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        masks = [m.numpy() for m in seg]

        # COCO format in albumentations is [x,y,w,h] non-normalized
        ih, iw, _ = img_in.shape
        bboxes_points = bboxes[:,1:]
        bboxes_points[:,0::2] *= iw
        bboxes_points[:,1::2] *= ih
        bbox_classes = bboxes[:,0]

        # albumentations expects "list" of keypoints e.g. [(x,y),(x,y),(x,y),(x,y)]
        keypoints_flat = torch.reshape(keypoints, (1,-1,3)).squeeze()
        keypoints_points = keypoints_flat[:,:2]
        keypoints_classes = keypoints_flat[:,2]

        transformed = self.transform(
            image = img_in,
            masks = masks,
            bboxes = bboxes_points,
            bbox_classes = bbox_classes,
            keypoints = keypoints_points,
            keypoints_classes= keypoints_classes,
        )
        transformed_image = transformed["image"]
        transformed_image = transformed_image.flip(-3) # to BGR format

        transformed_mask = torch.stack(transformed["masks"]) # stack list of masks

        transformed_bboxes = torch.tensor(transformed["bboxes"])
        transformed_bbox_classes = torch.tensor(transformed["bbox_classes"])
        # merge bboxes and classes back together
        transformed_bbox_classes = torch.unsqueeze(transformed_bbox_classes, dim=-1)
        out_bboxes = torch.cat((transformed_bbox_classes, transformed_bboxes), dim=1)

        transformed_keypoints = torch.tensor(transformed["keypoints"])
        transformed_keypoints_classes = torch.tensor(transformed["keypoints_classes"])
        # merge keypoints and classes back together
        transformed_keypoints_classes = torch.unsqueeze(transformed_keypoints_classes, dim=-1)
        out_keypoints = torch.cat((transformed_keypoints, transformed_keypoints_classes), dim=1)
        out_keypoints = torch.reshape(out_keypoints, (-1, keypoints.shape[1], keypoints.shape[2]))
        out_keypoints = mark_invisible_keypoints(out_keypoints, transformed_image)

        return transformed_image, classify, out_bboxes, transformed_mask, out_keypoints


class ValAugmentations:
    def __init__(self, image_size, normalize=True, to_tensor=True):
        self.image_size = image_size
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.transform = self.parse_cfg()

    def parse_cfg(self):
        # Only Resize (and optional Normalize/ToTensor) in validation
        augmentations = [A.Resize(self.image_size[0], self.image_size[1])]

        if self.normalize: augmentations.append(A.Normalize())
        if self.to_tensor: augmentations.append(A.pytorch.ToTensorV2())

        return A.Compose(
            augmentations,
            bbox_params=A.BboxParams(format="coco", label_fields=["bbox_classes"]),
            keypoint_params=A.KeypointParams(format="xy", label_fields=["keypoints_classes"], remove_invisible=False),
        )

    def __call__(self, data):
        img, classify, bboxes, seg, keypoints = data
        img_in = img.numpy()
        # albumentations expects with RGB image with HWC format
        img_in = np.transpose(img_in, (1,2,0))
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        masks = [m.numpy() for m in seg]

        # COCO format in albumentations is [x,y,w,h] non-normalized
        ih, iw, _ = img_in.shape
        bboxes_points = bboxes[:,1:]
        bboxes_points[:,0::2] *= iw
        bboxes_points[:,1::2] *= ih
        bbox_classes = bboxes[:,0]

        # if keypoints.shape[0]:
        #     show = img.numpy().transpose((1,2,0))
        #     for kp in keypoints:
        #         kp = kp[1:].numpy().reshape((-1,3))
        #         for i in range(len(kp)):
        #             cv2.circle(show, (round(kp[i,0]),round(kp[i,1])), 2, (255,0,0), 2)
        #     cv2.imshow('img', show)

        # albumentations expects "list" of keypoints e.g. [(x,y),(x,y),(x,y),(x,y)]
        keypoints_classes = keypoints[:,0]
        keypoints_flat = torch.reshape(keypoints[:,1:], (-1,3))
        keypoints_points = keypoints_flat[:,:2]
        keypoints_visibility = keypoints_flat[:,2]

        transformed = self.transform(
            image = img_in,
            masks = masks,
            bboxes = bboxes_points,
            bbox_classes = bbox_classes,
            keypoints = keypoints_points,
            keypoints_classes = keypoints_visibility, # not object class, but per-kp class
        )
        transformed_image = transformed["image"]
        transformed_image = transformed_image.flip(-3) # to BGR format

        transformed_mask = torch.stack(transformed["masks"]) # stack list of masks

        transformed_bboxes = torch.tensor(transformed["bboxes"])
        transformed_bbox_classes = torch.tensor(transformed["bbox_classes"])
        # merge bboxes and classes back together
        transformed_bbox_classes = torch.unsqueeze(transformed_bbox_classes, dim=-1)
        out_bboxes = torch.cat((transformed_bbox_classes, transformed_bboxes), dim=1)

        transformed_keypoints = torch.tensor(transformed["keypoints"])
        transformed_keypoints_classes = torch.tensor(transformed["keypoints_classes"])
        # merge keypoints and classes back together
        transformed_keypoints_classes = torch.unsqueeze(transformed_keypoints_classes, dim=-1)
        out_keypoints = torch.cat((transformed_keypoints, transformed_keypoints_classes), dim=1)

        out_keypoints = torch.reshape(out_keypoints, (-1, 3))
        out_keypoints = mark_invisible_keypoints(out_keypoints, transformed_image)
        out_keypoints = torch.reshape(out_keypoints, (keypoints.shape[0], keypoints.shape[1]-1))

        final_keypoints = torch.zeros_like(keypoints)
        final_keypoints[:,1:] = out_keypoints
        final_keypoints[:,1] = keypoints_classes

        # if final_keypoints.shape[0]:
        #     # print(f'final_keypoints {id}', final_keypoints)
        #     show = transformed_image.numpy().transpose((1,2,0))
        #     for kp in final_keypoints:
        #         kp = kp[1:].numpy().reshape((-1,3))
        #         for i in range(len(kp)):
        #             cv2.circle(show, (round(kp[i,0]),round(kp[i,1])), 2, (255,0,0), 2)
        #     cv2.imshow('new', show)
        #     cv2.waitKey(0)

        return transformed_image, classify, out_bboxes, transformed_mask, final_keypoints


def mark_invisible_keypoints(keypoints, image):
    # invisible keypoints should have label == 0
    _, h, w = image.shape
    # for dim in range(keypoints.shape[0]):
    # for kp in keypoints[dim]:
    for kp in keypoints:
        if not(0<=kp[0]<w and 0<=kp[1]<h):
            kp[2] = 0
    return keypoints
