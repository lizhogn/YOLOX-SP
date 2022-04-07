#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
import numpy as np

import torch
from scipy.optimize import linear_sum_assignment
from findmaxima2d import find_maxima, find_local_maxima 
import cv2

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def convert_mask_to_points(mask, up_factor=None):
    mask = mask * 255.0
    # filter out the low score pixel
    mask[mask < 60] = 0
    
    # find the local maxium point
    local_max = find_local_maxima(mask)
    y, x, out = find_maxima(mask, local_max, 10)

    points = np.stack([x, y], axis=1)
    if up_factor is not None:
        points = up_factor * points
    return points

def get_iou_matrix(gt_bx, dt_bx):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(gt_bx, 0)
    bb_test = np.expand_dims(dt_bx, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)

class CVATEvaluator:
    """
    CVAT dataset Evaluation
    """
    def __init__(
        self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        # detection
        total_correct = 0
        total_gt = 0
        total_dt = 0

        # endpoints
        points_recall = 0
        points_gt = 0

        for cur_iter, (imgs, bboxes, points) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                detections = postprocess(outputs[0], self.num_classes, self.confthre, self.nmsthre)
                masks = outputs[1].cpu().numpy().squeeze()

            # detections eval
            correct, dt_cnt, gt_cnt = self.detection_eval(detections, bboxes, iou_thre=0.5)
            total_correct += correct
            total_dt += dt_cnt
            total_gt += gt_cnt
            
            # endpoints eval
            pt_correct, pt_gt = self.mask_eval(masks, points)
            points_recall += pt_correct
            points_gt += pt_gt

        # detections
        det_precision = round(total_correct / (total_dt + 1e-10), 3)
        det_recall = round(total_correct / (total_gt + 1e-10), 3)

        # endpoints
        mask_recall = round(points_recall / (points_gt + 1e-10), 3)

        return {
            "detection": {
                "precision": det_precision,
                "recall": det_recall
            },
            "endpoints": mask_recall
        }
    

    def detection_eval(self, dt, gt, iou_thre=0.5):
        if dt[0] is None:
            return 0, 0, len(gt)
        else:
            dt = dt[0][:, 0:4].cpu().numpy()
            gt = gt[0][:, 0:4].numpy()
        correct = 0
        total_dt = len(dt)
        total_gt = len(gt)
        
        iou_matrix = get_iou_matrix(gt, dt)

        # Hungrian match
        x, y = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(x, y)))
        
        # filter out the matched with low IOU
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= iou_thre:
                correct += 1
        
        return correct, total_dt, total_gt

    def mask_eval(self, mask, gt, radius=5, input_size=(640, 640)):
        gt = gt[0].numpy()
        if len(gt) == 0:
            return 0, 0
        gt = np.concatenate([gt[:, :2], gt[:, 2:]], axis=0)

        mask = cv2.resize(mask, input_size)
        dt = convert_mask_to_points(mask)
        total_dt = len(dt)
        total_gt = len(gt)
        correct = 0

        gt_pts = np.expand_dims(np.stack(gt, axis=0), 0)
        dt_pts = np.expand_dims(np.stack(dt, axis=0), 1)
        dis_mat = np.sqrt(np.sum(np.square(gt_pts - dt_pts), axis=2))
        x, y = linear_sum_assignment(dis_mat)
        matched_indices = np.array(list(zip(x, y)))
        # filter out the matched indices
        for (x, y) in matched_indices:
            if dis_mat[x, y] <= radius:
                correct += 1
        return correct, total_gt
