#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .unet_head import HeatMapHead


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, det_head=None, mask_head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if det_head is None:
            det_head = YOLOXHead(80)
        if mask_head is None:
            mask_head = HeatMapHead(1)

        self.backbone = backbone
        self.det_head = det_head
        self.mask_head = mask_head

        # loss balancing
        params = torch.ones(2, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x, targets=None, target_mask=None):
        # fpn output content features of [stem, dark2, dark3, dark4, dark5]
        det_features, mask_features = self.backbone(x)

        if self.training:
            assert targets is not None
            det_loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.det_head(
                det_features, targets, x
            )
            mask_loss = self.mask_head(mask_features, target_mask)
            # mask_loss = torch.tensor([0])
            # total_loss = mask_loss
            # total_loss1 = det_loss + 10.0 * mask_loss
            # total_loss = torch.exp(-self.l_obj) * det_loss + torch.exp(-self.l_reg) * mask_loss + \
            #             (self.l_obj + self.l_reg)
            total_loss = 0.5 / (self.params[0] ** 2) * det_loss + torch.log(1 + self.params[0] ** 2) + \
                0.5 / (self.params[1] ** 2) * mask_loss + torch.log(1 + self.params[1] ** 2)
                
            # total_loss = det_loss
            loss_outputs = {
                "total_loss": total_loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss, 
                "cls_loss": cls_loss,
                "mask_loss": mask_loss,
                "num_fg": num_fg,
                # "obj_factor": self.l_obj,
                # "reg_factor": self.l_reg
            }
            return loss_outputs
        else:
            det_outputs = self.det_head(det_features)
            mask_outputs = self.mask_head(mask_features)
            
            return det_outputs, mask_outputs
