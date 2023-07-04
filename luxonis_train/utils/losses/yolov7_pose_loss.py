# Loss functions

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False,
             DIoU=False, CIoU=False, eps=1e-7) -> Tensor:  # type: ignore
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

class FocalLoss(nn.Module):
    # Source: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: Tensor, targets: Tensor):

        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1).float()
        targets = targets.view(-1).float().to(inputs.device)

        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE

        return focal_loss


class YoloV7PoseLoss(nn.Module):
    # Compute losses
    def __init__(self, model):
        super().__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.focal_loss = FocalLoss(alpha=h['fl_alpha'], gamma=h['fl_gamma'])

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        det = model  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = 0  # stride 16 index
        self.gr, self.hyp = model.gr, h
        self.nkpt = det.nkpt
        self.nc = det.nc
        self.na = det.na
        self.nl = det.nl
        self.anchors = det.anchors

    def __call__(self, kpt_pred, kpt, **_):
        # validation model output is kpt, features. The loss only needs features.
        if isinstance(kpt_pred, tuple):
            kpt_pred = kpt_pred[1]
        kpt_pred[0].shape[0]  # batch size
        device = kpt_pred[0].device
        lcls, lbox, lobj, lkpt, lkptv = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, tkpt, indices, anchors = self.build_targets(kpt_pred, kpt)  # targets
        kpt = kpt.to(device)

        # Losses
        for i, pi in enumerate(kpt_pred):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i].to(device)
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i].to(device), x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                #Direct kpt prediction
                pkpt_x = ps[:, 5 + self.nc::3] * 2. - 0.5
                pkpt_y = ps[:, 6 + self.nc::3] * 2. - 0.5
                pkpt_score = ps[:, 7 + self.nc::3]
                #mask
                tkpt[i] = tkpt[i].to(device)
                kpt_mask = (tkpt[i][:, 0::2] != 0)
                lkptv += self.BCEcls(pkpt_score, kpt_mask.float())
                d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2
                kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                lkpt += kpt_loss_factor * (torch.log(d + 1 + 1e-9) * kpt_mask).mean()
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:5+self.nc], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:5+self.nc], t)  # BCE


            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkptv *= self.hyp['cls']
        lkpt *= self.hyp['kpt']

        return (lbox + lobj + lcls + lkpt + lkptv).reshape([])

    def build_targets(self, p, targets):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        gain_length = 7 + 2 * self.nkpt
        gain = torch.ones(gain_length, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:gain_length-1] = torch.tensor(p[i].shape)[(2 + self.nkpt)*[3, 2]]
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, -1].long()  # anchor indices
            indices.append(
                (
                    b, a, gj.clamp_(0, gain[3].long() - 1),  # type: ignore
                    gi.clamp_(0, gain[2].long() - 1)  # type: ignore
                )
            )
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            for kpt in range(self.nkpt):
                t[:, 6 + 2 * kpt: 6 + 2 * (kpt + 1)][t[:, 6 + 2 * kpt: 6 + 2 * (kpt + 1)] != 0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]
            tkpt.append(t[:, 6:-1])
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, tkpt, indices, anch
