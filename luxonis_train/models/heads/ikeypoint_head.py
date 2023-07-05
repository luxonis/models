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
    anchor_grid: Tensor
    anchors: Tensor

    def __init__(self, n_classes, anchors,
                 n_keypoints, prev_out_shape,
                 inplace=True, connectivity=None,
                 **kwargs):
        super().__init__()
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.connectivity = connectivity  # for visualization
        self.type = KeyPointDetection()  # type: ignore
        ch = [prev[1] for prev in prev_out_shape]
        self.gr = 1.0  # TODO: find out what this is
        self.n_classes = n_classes
        self.n_keypoints = n_keypoints
        self.no_det= n_classes + 5  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.n_keypoints  # number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        self.m_kpt = nn.ModuleList(
            nn.Sequential(
                YoloV7DWConv(x, x, k=3), YoloV7Conv(x,x),
                YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                YoloV7DWConv(x, x, k=3), YoloV7Conv(x,x),
                YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                YoloV7DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1))
            for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, inputs):
        z = []  # inference output
        x = []  # layer outputs

        for i in range(self.nl):
            x.append(torch.cat(
                (self.im[i](self.m[i](self.ia[i](inputs[i]))),
                 self.m_kpt[i](inputs[i])), axis=1))  # type: ignore

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx
                             ).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :5 + self.n_classes]
            x_kpt = x[i][..., 5 + self.n_classes:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.n_keypoints == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(
                        1, self.na, 1, 1, 2) # wh
                    x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(
                        1,1,1,1,self.n_keypoints)) * self.stride[i]  # xy
                    x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(
                        1,1,1,1,self.n_keypoints)) * self.stride[i]  # xy
                    x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

                # * I guess this can be deleted
                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat(
                        (1, 1, 1, 1, self.n_keypoints))) * self.stride[i]  # xy
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
