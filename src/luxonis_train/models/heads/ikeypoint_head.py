#
# Adapted from: https://github.com/WongKinYiu/yolov7
# License: https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md
#

import torch
import torch.nn as nn
import math
from typing import List
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes, draw_keypoints

from luxonis_train.models.heads.base_heads import BaseHead
from luxonis_train.models.modules import ConvModule, autopad
from luxonis_train.utils.constants import HeadType, LabelType
from luxonis_train.utils.boxutils import non_max_suppression_kpts


class IKeypoint(BaseHead):
    head_types: List[HeadType] = [
        HeadType.OBJECT_DETECTION,
        HeadType.KEYPOINT_DETECTION,
    ]
    label_types: List[LabelType] = [LabelType.BOUNDINGBOX, LabelType.KEYPOINT]

    def __init__(
        self,
        n_classes: int,
        prev_out_shapes: list,
        original_in_shape: list,
        n_keypoints: int,
        anchors: list,
        attach_index: int = -1,
        main_metric: str = "map",
        connectivity: list = None,
        **kwargs,
    ):
        """IKeypoint head which is used for object and keypoint detection

        Args:
            n_classes (int): Number of classes
            prev_out_shapes (list): List of shapes of previous outputs
            original_in_shape (list): Original input shape to the model
            n_keypoints (int): Number of keypoints
            anchors (list): Anchors used for object detection
            attach_index (int, optional): Index of previous output that the head attaches to. Defaults to -1.
            main_metric (str, optional): Name of the main metric which is used for tracking training process. Defaults to "map".
            connectivity (list, optional): Connectivity mapping used in visualization. Defaults to None.
        """
        super().__init__(
            n_classes=n_classes,
            prev_out_shapes=prev_out_shapes,
            original_in_shape=original_in_shape,
            attach_index=attach_index,
        )

        self.main_metric: str = main_metric

        self.n_keypoints = n_keypoints
        self.connectivity = connectivity

        ch = [prev[1] for prev in self.prev_out_shapes]
        self.gr = 1.0  # TODO: find out what this is
        self.no_det = n_classes + 5  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.n_keypoints  # number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.anchors = a  # shape(nl,na,2)
        self.anchor_grid = a.clone().view(self.nl, 1, -1, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        self.m_kpt = nn.ModuleList(
            nn.Sequential(
                ConvModule(
                    x,
                    x,
                    kernel_size=3,
                    padding=autopad(3),
                    groups=math.gcd(x, x),
                    activation=nn.SiLU(),
                ),
                ConvModule(
                    x, x, kernel_size=1, padding=autopad(1), activation=nn.SiLU()
                ),
                ConvModule(
                    x,
                    x,
                    kernel_size=3,
                    padding=autopad(3),
                    groups=math.gcd(x, x),
                    activation=nn.SiLU(),
                ),
                ConvModule(
                    x, x, kernel_size=1, padding=autopad(1), activation=nn.SiLU()
                ),
                ConvModule(
                    x,
                    x,
                    kernel_size=3,
                    padding=autopad(3),
                    groups=math.gcd(x, x),
                    activation=nn.SiLU(),
                ),
                ConvModule(
                    x, x, kernel_size=1, padding=autopad(1), activation=nn.SiLU()
                ),
                ConvModule(
                    x,
                    x,
                    kernel_size=3,
                    padding=autopad(3),
                    groups=math.gcd(x, x),
                    activation=nn.SiLU(),
                ),
                ConvModule(
                    x, x, kernel_size=1, padding=autopad(1), activation=nn.SiLU()
                ),
                ConvModule(
                    x,
                    x,
                    kernel_size=3,
                    padding=autopad(3),
                    groups=math.gcd(x, x),
                    activation=nn.SiLU(),
                ),
                ConvModule(
                    x, x, kernel_size=1, padding=autopad(1), activation=nn.SiLU()
                ),
                ConvModule(
                    x,
                    x,
                    kernel_size=3,
                    padding=autopad(3),
                    groups=math.gcd(x, x),
                    activation=nn.SiLU(),
                ),
                nn.Conv2d(x, self.no_kpt * self.na, 1),
            )
            for x in ch
        )

        self.stride = torch.tensor(
            [self.original_in_shape[2] / x[2] for x in self.prev_out_shapes]
        )
        self.anchors /= self.stride.view(-1, 1, 1)
        self._check_anchor_order()

    def forward(self, inputs):
        z = []  # inference output
        x = []  # layer outputs

        if self.anchor_grid.device != inputs[0].device:
            self.anchor_grid = self.anchor_grid.to(inputs[0].device)

        for i in range(self.nl):
            x.append(
                torch.cat(
                    (
                        self.im[i](self.m[i](self.ia[i](inputs[i]))),
                        self.m_kpt[i](inputs[i]),
                    ),
                    axis=1,
                )
            )  # type: ignore

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = (
                x[i]
                .view(bs, self.na, self.no, ny, nx)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            x_det = x[i][..., : 5 + self.n_classes]
            x_kpt = x[i][..., 5 + self.n_classes :]

            # from this point down only needed for inference
            if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
            kpt_grid_x = self.grid[i][..., 0:1]
            kpt_grid_y = self.grid[i][..., 1:2]

            if self.n_keypoints == 0:
                y = x[i].sigmoid()
            else:
                y = x_det.sigmoid()

            xy = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                1, self.na, 1, 1, 2
            )  # wh
            x_kpt[..., 0::3] = (
                x_kpt[..., ::3] * 2.0
                - 0.5
                + kpt_grid_x.repeat(1, 1, 1, 1, self.n_keypoints)
            ) * self.stride[
                i
            ]  # xy
            x_kpt[..., 1::3] = (
                x_kpt[..., 1::3] * 2.0
                - 0.5
                + kpt_grid_y.repeat(1, 1, 1, 1, self.n_keypoints)
            ) * self.stride[
                i
            ]  # xy
            x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

            y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)
            z.append(y.view(bs, -1, self.no))

        # returns Tuple[kpt, features]
        return torch.cat(z, 1), x

    def postprocess_for_loss(self, output: tuple, label_dict: dict):
        kpts = label_dict[LabelType.KEYPOINT]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        label = torch.zeros((len(boxes), nkpts * 2 + 6))
        label[:, :2] = boxes[:, :2]
        label[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "cxcywh")
        label[:, 6::2] = kpts[:, 2::3]  # insert kp x coordinates
        label[:, 7::2] = kpts[:, 3::3]  # insert kp y coordinates
        return output, label

    def postprocess_for_metric(self, output: tuple, label_dict: dict):
        kpts = label_dict[LabelType.KEYPOINT]
        boxes = label_dict[LabelType.BOUNDINGBOX]
        nkpts = (kpts.shape[1] - 2) // 3
        label = torch.zeros((len(boxes), nkpts * 2 + 6))
        label[:, :2] = boxes[:, :2]
        label[:, 2:6] = box_convert(boxes[:, 2:], "xywh", "cxcywh")
        label[:, 6::2] = kpts[:, 2::3]  # insert kp x coordinates
        label[:, 7::2] = kpts[:, 3::3]  # insert kp y coordinates

        nms = non_max_suppression_kpts(output[0])
        output_list_map = []
        label_list_map = []
        image_size = self.original_in_shape[2:]
        for i in range(len(nms)):
            output_list_map.append(
                {
                    "boxes": nms[i][:, :4],
                    "scores": nms[i][:, 4],
                    "labels": nms[i][:, 5].int(),
                }
            )

            curr_label = label[label[:, 0] == i].to(nms[i].device)
            curr_bboxs = box_convert(curr_label[:, 2:6], "cxcywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list_map.append(
                {
                    "boxes": curr_bboxs,
                    "labels": curr_label[:, 1].int(),
                }
            )

        output_list_oks, label_list_oks = (
            [],
            [],
        )  # TODO: implement oks and add correct output and labels

        # metric mapping is needed here because each metrics requires different output/label format
        metric_mapping = {"map": 0, "oks": 1}
        return (
            (output_list_map, output_list_oks),
            (label_list_map, label_list_oks),
            metric_mapping,
        )

    def draw_output_to_img(self, img: torch.Tensor, output: tuple, idx: int):
        curr_output = output[0][idx]
        nms = non_max_suppression_kpts(
            curr_output.unsqueeze(0), conf_thresh=0.25, iou_thresh=0.45
        )[0]
        bboxes = nms[:, :4]
        img = draw_bounding_boxes(img, bboxes)
        kpts = nms[:, 6:].reshape(-1, self.n_keypoints, 3)
        img = draw_keypoints(img, kpts, colors="red", connectivity=self.connectivity)
        return img

    def get_output_names(self, idx: int):
        # TODO: check if this is correct output name
        return f"output{idx}"

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _check_anchor_order(self):
        a = self.anchor_grid.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = self.stride[-1] - self.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            print("Reversing anchor order")
            self.anchors[:] = self.anchors.flip(0)
            self.anchor_grid[:] = self.anchor_grid.flip(0)


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=0.02)

    def forward(self, x):
        return self.implicit.expand_as(x) + x


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1.0, std=0.02)

    def forward(self, x):
        return self.implicit.expand_as(x) * x
