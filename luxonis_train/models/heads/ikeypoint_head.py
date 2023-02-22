#
# Soure:
# License:
#

import torch
import torch.nn as nn

from luxonis_train.utils.head_type import KeyPointDetection
from luxonis_train.models.modules import YoloV7Conv, YoloV7DWConv, ImplicitA, ImplicitM

class IKeypoint(nn.Module):

    def __init__(self, prev_out_shape, n_classes, n_keypoints, n_layers=3, n_anchors=3, inplace=True, dw_conv_kpt=True, **kwargs):
        super(IKeypoint, self).__init__()

        ch = [shape[1] for shape in prev_out_shape]

        self.n_classes = n_classes  # number of classes
        self.n_keypoints = n_keypoints
        self.type = KeyPointDetection()

        self.dw_conv_kpt = dw_conv_kpt
        self.no_det=(self.n_classes + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3*self.n_keypoints ## number of outputs per anchor for keypoints
        self.no = self.no_det+self.no_kpt
        self.nl = n_layers  # number of detection layers
        self.na = n_anchors  # number of anchors

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        if self.n_keypoints is not None:
            if self.dw_conv_kpt: #keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                            nn.Sequential(YoloV7DWConv(x, x, k=3), YoloV7Conv(x,x),
                                          YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                                          YoloV7DWConv(x, x, k=3), YoloV7Conv(x,x),
                                          YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                                          YoloV7DWConv(x, x, k=3), YoloV7Conv(x, x),
                                          YoloV7DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else: #keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        if self.n_keypoints is None or self.n_keypoints==0:
            x[-1] = self.im[-1](self.m[-1](self.ia[-1](x[-1])))  # conv
        else :
            x[-1] = torch.cat((self.im[-1](self.m[-1](self.ia[-1](x[-1]))), self.m_kpt[-1](x[-1])), axis=1)

        bs, _, ny, nx = x[-1].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x[-1] = x[-1].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        x_det = x[-1][..., :6]
        x_kpt = x[-1][..., 6:]

        if self.grid[-1].shape[2:4] != x[-1].shape[2:4]:
            self.grid[-1] = self._make_grid(nx, ny).to(x[-1].device)
        kpt_grid_x = self.grid[-1][..., 0:1]
        kpt_grid_y = self.grid[-1][..., 1:2]

        if self.n_keypoints == 0:
            y = x[-1].sigmoid()
        else:
            y = x_det.sigmoid()

        if self.inplace:
            xy = (y[..., 0:2] * 2. - 0.5 + self.grid[-1]) * self.stride  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid.view(1, self.na, 1, 1, 2) # wh
            if self.n_keypoints != 0:
                x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride # xy
                x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride  # xy
                x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

            y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim = -1)

        else:
            xy = (y[..., 0:2] * 2. - 0.5 + self.grid[-1]) * self.stride[-1]  # xy
            wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            if self.n_keypoints != 0:
                y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[-1].repeat((1,1,1,1,self.n_keypoints))) * self.stride  # xy
            y = torch.cat((xy, wh, y[..., 4:]), -1)

        z.append(y.view(bs, -1, self.no))

        return [x[-1], torch.cat(z, 1)]

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
