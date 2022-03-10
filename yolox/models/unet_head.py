import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou
from yolox.utils.boxes import xyxy2xywh

from .losses import IOUloss, FocalLoss
from torch.nn import MSELoss
from .network_blocks import BaseConv, DWConv

class HeatMapHead(nn.Module):
    def __init__(
        self,
        num_class=1,
        in_channel=64
    ):
        super().__init__()
        self.outc = OutConv(in_channel, num_class)
        # self.focal_loss = FocalLoss()
    
    def forward(self, xin, target_mask=None):
        preds = self.outc(xin)
        if target_mask is None:
            return preds
        else:
            preds = preds[:, 0, :, :]
            heatmap = target_mask[:, 0, :, :]
            mask = target_mask[:, 1, :, :]
            loss = self.get_loss(preds, heatmap, mask)
            return loss
    
    def get_loss(self, preds, heatmap, mask):
        loss = torch.pow(preds-heatmap, 2) * mask
        return loss.sum()

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # self.activate = nn.LeakyReLU()
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.activate(x)
        return x