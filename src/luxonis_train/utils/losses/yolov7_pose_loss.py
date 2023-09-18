#
# Adapted from: https://github.com/WongKinYiu/yolov7
# License: https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md
#

import torch
import torch.nn as nn

from luxonis_train.utils.losses.base_loss import BaseLoss
from luxonis_train.utils.boxutils import bbox_iou
from luxonis_train.utils.losses.common import BCEWithLogitsLoss


class YoloV7PoseLoss(BaseLoss):
    def __init__(
        self,
        cls_pw=1.0,
        obj_pw=1.0,
        gamma=2,
        alpha=0.25,
        label_smoothing=0.0,
        iou_ratio=1,
        box_weight=0.05,
        kpt_weight=0.10,
        kptv_weight=0.6,
        cls_weight=0.6,
        obj_weight=0.7,
        anchor_t=4.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.n_classes = self.head_attributes.get("n_classes")
        self.n_keypoints = self.head_attributes.get("n_keypoints")
        self.n_anchors = self.head_attributes.get("n_anchors")
        self.num_heads = self.head_attributes.get("num_heads")
        self.anchors = self.head_attributes.get("anchors")
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.num_heads, [4.0, 1.0, 0.25, 0.06, 0.02]
        )

        self.iou_ratio = iou_ratio
        self.box_weight = box_weight
        self.kpt_weight = kpt_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.kptv_weight = kptv_weight
        self.anchor_t = anchor_t

        self.BCEcls = BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw]))
        self.BCEobj = BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))

        # Class label smoothing targets (https://arxiv.org/pdf/1902.04103.pdf eqn 3)
        self.positive_smooth_const, self.negative_smooth_const = self._smooth_BCE(
            eps=label_smoothing
        )

    def forward(self, kpt_pred, kpt, epoch, step):
        # model output is (kpt, features). The loss only needs features.
        kpt_pred = kpt_pred[1]
        kpt_pred[0].shape[0]  # batch size
        device = kpt_pred[0].device
        lcls, lbox, lobj, lkpt, lkptv = [
            torch.zeros(1, device=device) for _ in range(5)
        ]
        tcls, tbox, tkpt, indices, anchors = self.build_targets(kpt_pred, kpt)
        kpt = kpt.to(device)

        # Losses
        for i, pi in enumerate(kpt_pred):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i].to(device)
                pbox = torch.cat((pxy, pwh), 1)

                iou = bbox_iou(
                    pbox,
                    tbox[i].to(device),
                    box_format="xywh",
                    iou_type="ciou",
                    element_wise=True,
                )

                lbox += (1.0 - iou).mean()  # iou loss
                # Direct kpt prediction
                pkpt_x = ps[:, 5 + self.n_classes :: 3] * 2.0 - 0.5
                pkpt_y = ps[:, 6 + self.n_classes :: 3] * 2.0 - 0.5
                pkpt_score = ps[:, 7 + self.n_classes :: 3]
                # mask
                tkpt[i] = tkpt[i].to(device)
                kpt_mask = tkpt[i][:, 0::2] != 0
                lkptv += self.BCEcls(pkpt_score, kpt_mask.float(), epoch, step)[0]
                d = (pkpt_x - tkpt[i][:, 0::2]) ** 2 + (pkpt_y - tkpt[i][:, 1::2]) ** 2
                kpt_loss_factor = (
                    torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0)
                ) / (torch.sum(kpt_mask != 0) + 1e-9)
                lkpt += kpt_loss_factor * (torch.log(d + 1 + 1e-9) * kpt_mask).mean()
                # Objectness
                tobj[b, a, gj, gi] = (
                    1.0 - self.iou_ratio
                ) + self.iou_ratio * iou.detach().clamp(0).type(tobj.dtype)

                # Classification
                if self.n_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(
                        ps[:, 5 : 5 + self.n_classes],
                        self.negative_smooth_const,
                        device=device,
                    )
                    t[range(n), tcls[i]] = self.positive_smooth_const
                    lcls += self.BCEcls(ps[:, 5 : 5 + self.n_classes], t, epoch, step)[
                        0
                    ]  # BCE

            obji = self.BCEobj(pi[..., 4], tobj, epoch, step)[0]
            lobj += obji * self.balance[i]  # obj loss
        lbox *= self.box_weight
        lobj *= self.obj_weight
        lcls *= self.cls_weight
        lkptv *= self.kptv_weight
        lkpt *= self.kpt_weight

        loss = (lbox + lobj + lcls + lkpt + lkptv).reshape([])

        sub_losses = {
            "lbox": lbox.detach(),
            "lobj": lobj.detach(),
            "lcls": lcls.detach(),
            "lkptv": lkptv.detach(),
            "lkpt": lkpt.detach(),
        }

        return loss, sub_losses

    def build_targets(self, p, targets):
        n_anchors, nt = self.n_anchors, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        gain_length = 7 + 2 * self.n_keypoints
        gain = torch.ones(gain_length, device=targets.device)
        ai = (
            torch.arange(n_anchors, device=targets.device)
            .float()
            .view(n_anchors, 1)
            .repeat(1, nt)
        )
        targets = torch.cat((targets.repeat(n_anchors, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.num_heads):
            anchors = self.anchors[i]
            gain[2 : gain_length - 1] = torch.tensor(p[i].shape)[
                (2 + self.n_keypoints) * [3, 2]
            ]
            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
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
                    b,
                    a,
                    gj.clamp_(0, gain[3].long() - 1),  # type: ignore
                    gi.clamp_(0, gain[2].long() - 1),  # type: ignore
                )
            )
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            for kpt in range(self.n_keypoints):
                low = 6 + 2 * kpt
                high = 6 + 2 * (kpt + 1)
                t[:, low:high][t[:, low:high] != 0] -= gij[t[:, low:high] != 0]
            tkpt.append(t[:, 6:-1])
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, tkpt, indices, anch

    def _smooth_BCE(self, eps=0.1):
        """Returns positive and negative label smoothing BCE targets
        Source: https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
        """
        return 1.0 - 0.5 * eps, 0.5 * eps
