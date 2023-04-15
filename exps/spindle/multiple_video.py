#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os
import torch


import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from yolox.models.unet_head import HeatMapHead
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
        self.train_img_dir = [
            "datasets/S_pombe/each_task/task1/images",
            "datasets/S_pombe/each_task/task3/images",
            "datasets/S_pombe/each_task/task5/images",
        ]
        self.train_anno_path = [
            "datasets/S_pombe/each_task/task1/annotations.xml",
            "datasets/S_pombe/each_task/task3/annotations.xml",
            "datasets/S_pombe/each_task/task5/annotations.xml",
        ]
        self.val_img_dir = [
            "datasets/S_pombe/each_task/task2/images",
            "datasets/S_pombe/each_task/task4/images",
            "datasets/S_pombe/each_task/task6/images",
        ]
        self.val_anno_path = [
            "datasets/S_pombe/each_task/task2/annotations.xml",
            "datasets/S_pombe/each_task/task4/annotations.xml",
            "datasets/S_pombe/each_task/task6/annotations.xml",
        ]

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
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
        self.max_epoch = 150
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.001
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 1
        self.eval_interval = 1
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.2
        self.nmsthre = 0.3

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            CVATVideoDataset,
            TrainTransform,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            self.dataset = CVATVideoDataset(        
                imgs_dir_list=self.train_img_dir,
                anno_path_list=self.train_anno_path,
                img_size=(640, 640),
                preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=0.5,
                        hsv_prob=0),
                mosaic=True,
                mode="train"    # "train" or "eval"
            )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        dataloader_kwargs = {"batch_size": batch_size, 
                            "num_workers": self.data_num_workers, 
                            "pin_memory": True,
                            "shuffle": True}

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import CVATVideoDataset, ValTransform

        valdataset = CVATVideoDataset(
            imgs_dir_list=self.val_img_dir,
            anno_path_list=self.val_anno_path,
            img_size=(640, 640),
            preproc=ValTransform(legacy=legacy),
            mosaic=False,
            mode="eval"    # "train" or "eval"
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = 1
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
