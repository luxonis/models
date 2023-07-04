import torch
from torch import Tensor, nn

from luxonis_train.models.modules.common import (
    ImplicitA,
    ImplicitM,
    YoloV7Conv,
    YoloV7DWConv,
)
from luxonis_train.utils.head_type import KeyPointDetection


class IKeypoint(nn.Module):
    stride: Tensor  # strides computed during build

    def __init__(self, n_classes, anchors,
                 n_keypoints, prev_out_shape, inplace=True, **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.hyp = {
            "lr0": 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": 0.01,  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": 0.937,  # SGD momentum/Adam beta1
            "weight_decay": 0.0005,  # optimizer weight decay 5e-4
            "warmup_epochs": 3.0,  # warmup epochs (fractions ok)
            "warmup_momentum": 0.8,  # warmup initial momentum
            "warmup_bias_lr": 0.1,  # warmup initial bias lr
            "box": 0.05,  # box loss gain
            "kpt": 0.10,  # kpt loss gain
            "cls": 0.6,  # cls loss gain
            "cls_pw": 1.0,  # cls BCELoss positive_weight
            "obj": 0.7,  # obj loss gain (scale with pixels)
            "obj_pw": 1.0,  # obj BCELoss positive_weight
            "iou_t": 0.20,  # IoU training threshold
            "anchor_t": 4.0,  # anchor-multiple threshold
            # "anchors": 3  # anchors per output layer (0 to ignore)
            "fl_alpha": 0.25,  # focal loss alpha
            "fl_gamma": 2.0,  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
            "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
            "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
            "degrees": 0.0,  # image rotation (+/- deg)
            "translate": 0.1,  # image translation (+/- fraction)
            "scale": 0.5,  # image scale (+/- gain)
            "shear": 0.0,  # image shear (+/- deg)
            "perspective": 0.0,  # image perspective (+/- fraction), range 0-0.001
            "flipud": 0.0,  # image flip up-down (probability)
            "fliplr": 0.5,  # image flip left-right (probability)
            "mosaic": 0.0,  # image mosaic (probability)
            "mixup": 0.0,  # image mixup (probability)
        }
        self.type = KeyPointDetection()
        ch = [prev[1] for prev in prev_out_shape]
        self.gr = 1.0  # TODO: find out what this is
        self.nc = self.n_classes = n_classes  # number of classes
        self.nkpt = self.n_keypoints = n_keypoints
        self.no_det=(n_classes + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt ## number of outputs per anchor for keypoints
        self.no = self.no_det+self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        self.m_kpt = nn.ModuleList(
                    nn.Sequential(
                        YoloV7DWConv(x, x, k=3), YoloV7Conv(x,x),
                        YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                        YoloV7DWConv(x, x, k=3), YoloV7Conv(x,x),
                        YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                        YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                        YoloV7DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, xx):
        z = []  # inference output
        x = [None] * self.nl  # layer outputs

        for i in range(self.nl):
            x[i] = torch.cat(
                (self.im[i](self.m[i](self.ia[i](xx[i]))),
                 self.m_kpt[i](xx[i])), axis=1)  # type: ignore

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :5+self.nc]
            x_kpt = x[i][..., 5+self.nc:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2) # wh
                    x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
                    x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,self.nkpt)) * self.stride[i]  # xy
                    x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1,1,1,1,self.nkpt))) * self.stride[i]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        if self.training:
            return x
        else:
            return torch.cat(z, 1), x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def check_anchor_order(self):
        a = self.anchor_grid.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = self.stride[-1] - self.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            print('Reversing anchor order')
            self.anchors[:] = self.anchors.flip(0)
            self.anchor_grid[:] = self.anchor_grid.flip(0)
