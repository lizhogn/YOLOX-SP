import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou
from yolox.utils.boxes import xyxy2xywh

from .losses import IOUloss
from torch.nn import MSELoss
from .network_blocks import BaseConv, DWConv


class UnetHead(nn.Module):
    def __init__(
        self,
        num_class,
        width=1,
        strides=[2, 4, 8, 16, 32],
        in_channels=[64, 128, 256, 512, 1024]
    ):
        """
        Args:
            act(str): acitvation type of conv. Default value is "silu"
            depthwise (bool): whether apply depthwise conv in conv branch. Default value: False
        """
        super().__init__()

        self.num_class = num_class
        self.width = width
        self.strides = strides
        self.in_channels = in_channels 

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outc = OutConv(64, self.num_class)

    def forward(self, xin, target_mask=None):
        """xin is the output of [stem, dark2, dark3, dark4, dark5]"""
        [stem, dark2, dark3, dark4, dark5] = xin
        x = self.up1(dark5, dark4)
        x = self.up2(x, dark3)
        x = self.up3(x, dark2)
        x = self.up4(x, stem)
        x = self.up5(x)
        preds = self.outc(x)
        if target_mask is None:
            return preds
        else:
            loss = self.get_losses(preds, target_mask)
            return loss

    def get_losses(self, preds, targets):
        mask_loss = MSELoss(preds, targets)
        return mask_loss

class HeatMapHead(nn.Module):
    def __init__(
        self,
        num_class=1,
        in_channel=64
    ):
        super().__init__()
        self.outc = OutConv(in_channel, num_class)
        self.mse = MSELoss(reduction="mean")
    
    def forward(self, xin, target_mask=None):
        pred_mask = self.outc(xin)
        if target_mask is None:
            return pred_mask
        else:
            loss = self.get_losses(pred_mask, target_mask)
            return loss
    
    def get_losses(self, preds, targets):
        mask_loss = self.mse(preds, targets)
        return mask_loss


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x