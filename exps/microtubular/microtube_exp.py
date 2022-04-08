#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import torch

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.act = 'silu'

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 0
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        self.train_img_dir = "/home/zhognli/YOLOX/datasets/spindle/train_set/images"
        self.train_anno_path = "/home/zhognli/YOLOX/datasets/spindle/train_set/annotations.xml"
        self.val_img_dir = "/home/zhognli/YOLOX/datasets/spindle/test_set/images"
        self.val_anno_path = "/home/zhognli/YOLOX/datasets/spindle/test_set/annotations.xml"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0
        self.mixup_prob = 0
        self.hsv_prob = 0
        self.del_green_prob = 0.5
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.7, 1.3)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 2.0
        self.enable_mosaic = True
        self.enable_mixup = False

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 300
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.2
        self.nmsthre = 0.65
