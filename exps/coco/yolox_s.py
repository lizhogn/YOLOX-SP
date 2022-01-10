#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "/home/zhognli/YOLOX/datasets/tiny-coco/small_coco"
        self.train_ann = "/home/zhognli/YOLOX/datasets/tiny-coco/small_coco/instances_train2017_small.json"
        self.val_ann = "/home/zhognli/YOLOX/datasets/tiny-coco/small_coco/instances_train2017_small.json"
        self.train_img_name = "train_2017_small"
        self.val_img_name   = "train_2017_small"

        self.num_classes = 80

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
