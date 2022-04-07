import os
import os.path
import pickle
import xml.etree.ElementTree as ET
from loguru import logger

import cv2
import numpy as np
import random
import math
from yolox.data.datasets.datasets_wrapper import Dataset
import matplotlib.pyplot as plt

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset

class CVATVideoDataset(Dataset):
    """
    CVAT Video Dataset

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    def __init__(
        self,
        img_dir,
        anno_path,
        img_size=(640, 640),
        preproc=None,
        mosaic=False
        # target_transform=AnnotationTransform()
    ):
        super().__init__(img_size)
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.img_size = img_size    # (img_h, img_w)
        self.mask_scale = 4
        self.preproc = preproc
        self.mosaic = mosaic
        
        # load the annotation data
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """parser the xml annotation file
        Output annotation format:
            {
                "0": {
                    "img_path": "path/to/img",
                    "bboxes": [lists of bboxes],
                    "points": [lists of points],
                    "bboxes_id": [list of bboxes identity],
                    "points_id": [list of points identity]
                },
                "1": {
                    ...
                },...
            }

        """
        # step1: load the xml file
        xml_root = ET.parse(self.anno_path).getroot()
        anno_data = {}

        # step2: parse the xml file
        for obj in xml_root.iter("track"):
            track_id = obj.attrib["id"]
            track_label = obj.attrib["label"]
            if track_label == "microtubule":
                iter_label = "box"
            elif track_label == "pole":
                iter_label = "points"
            else:
                continue
            for frame in obj.iter(iter_label):
                frame_id = frame.attrib["frame"]
                img_name = "frame_{:0>6d}.PNG".format(int(frame_id))
                is_outside = True if frame.attrib["outside"] == "1" else False
                
                img_path = os.path.join(self.img_dir, img_name)
                if not os.path.exists(img_path):
                    continue

                # save anno
                if not is_outside:
                    if frame_id not in anno_data:
                        anno_data[frame_id] = {
                            "img_name": img_name,
                            "bboxes": [],
                            "points": [],
                            "bboxes_id": [],
                            "points_id": []
                        }
                    if iter_label == "box":
                        # bboxes
                        x1, y1 = int(float(frame.attrib["xtl"])), int(float(frame.attrib["ytl"]))
                        x2, y2 = int(float(frame.attrib["xbr"])), int(float(frame.attrib["ybr"]))
                        anno_data[frame_id]["bboxes"].append([x1, y1, x2, y2])
                        anno_data[frame_id]["bboxes_id"].append(track_id)
                    else:
                        # points
                        points = frame.attrib["points"].split(";")
                        if len(points) < 2:
                            continue
                        x1, y1 = [int(float(x)) for x in points[0].split(",")]
                        x2, y2 = [int(float(x)) for x in points[1].split(",")]
                        anno_data[frame_id]["points"].append([x1, y1, x2, y2])
                        anno_data[frame_id]["points_id"].append(track_id)
        
        # step3: sort the dict by the key
        anno_data = dict(sorted(anno_data.items(), key=lambda x: int(x[0])))

        return list(anno_data.values())

    def __len__(self):
        return len(self.annotations)

    def pull_item(self, idx):
        cur_anno = self.annotations[idx]
        img_name = cur_anno["img_name"]
        img_path = os.path.join(self.img_dir, img_name)
        
        # load the img
        img, hw_origin, hw_resize = self.load_image(img_path)
        
        # load bboxes
        bboxes = np.asarray(cur_anno["bboxes"], dtype=np.float32)
        bboxes = self._norm_bboxes(bboxes, old_size=hw_origin, new_size=hw_resize)
        bboxes = np.concatenate([bboxes, np.zeros(shape=(len(bboxes), 1), dtype=np.float32)], axis=1)

        # load points
        points = np.asarray(cur_anno["points"], dtype=np.float32)
        points = self._norm_bboxes(points, old_size=hw_origin, new_size=hw_resize)
        mask = self._mask_generate(bboxes, points, hw_resize, scale=self.mask_scale)

        return img, bboxes, mask
        
    def __getitem__(self, idx):
        '''
        Returns: (before preproc)
            img (numpy.ndarray): resized image
                The shape is: [img_h, img_w, 3], values range from 0 to 255 
            mask (numpy.ndarray): mask for segmentation task
                The shape is: [img_h, img_w, 1], values range from 0.0 to 1.0
            bboxes (numpy.ndarray): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [x1, y1, x2, y2, class]:
                    class (float): class index. start from 0
                    x1, y1 (float) : top-left points whose values range from 0 to 640
                    x2, y2 (float) : right-down points whose values range from 0 to 640
            points (numpy.ndarray): endpoints data
                The shape is : [max_points, 4]
                each point consists of [x1, y1, x2, y2], (at new size image scale)
        '''

        # pull item
        if self.mosaic:
            img, bboxes, mask = self.mosaic_generate(idx)
        else:
            img, bboxes, mask = self.pull_item(idx)

        if self.preproc is not None:
            # concat the image and mask together
            img, mask, bboxes = self.preproc(img, bboxes, self.input_dim, mask)

        return img, mask, bboxes
    
    def mosaic_generate(self, idx):
        mosaic_labels = []
        input_h, input_w = self.input_dim

        # mosaic center x, y
        yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

        # 3 additional image indices
        indices = [idx] + [random.randint(0, len(self.annotations) - 1) for _ in range(3)]

        for i_mosaic, index in enumerate(indices):
            img, bboxes, mask = self.pull_item(index)

            # generate output mosaic image
            (h, w, c) = img.shape[:3]
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 0, dtype=np.uint8)
                mosaic_mask = np.full((input_h * 2 // self.mask_scale, input_w * 2 // self.mask_scale, 2), 0, dtype=np.uint8)

            (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = self.get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )
            (lm_x1, lm_y1, lm_x2, lm_y2) = [x // self.mask_scale for x in (l_x1, l_y1, l_x2, l_y2)]
            (sm_x1, sm_y1, sm_x2, sm_y2) = [x // self.mask_scale for x in (s_x1, s_y1, s_x2, s_y2)]
            lm_x2, lm_y2 = lm_x1 + (sm_x2 - sm_x1), lm_y1 + (sm_y2 - sm_y1)

            mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
            mosaic_mask[lm_y1:lm_y2, lm_x1:lm_x2] = mask[sm_y1:sm_y2, sm_x1:sm_x2]

            padw, padh = l_x1 - s_x1, l_y1 - s_y1
            labels = bboxes.copy()
            if labels.size > 0:
                labels[:, 0] = labels[:, 0] + padw
                labels[:, 1] = labels[:, 1] + padh
                labels[:, 2] = labels[:, 2] + padw
                labels[:, 3] = labels[:, 3] + padh
            
            mosaic_labels.append(labels)
        
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

        mosaic_img, mosaic_mask, mosaic_labels = self.random_affine(
            mosaic_img,
            mosaic_mask,
            mosaic_labels,
            target_size=(input_w, input_h),
            mask_scale=self.mask_scale,
            degrees=10.0,
            translate=0.1,
            scales=(0.5, 1.5),
            shear=2.0,
        )
        
        # filter out small box
        labels_w = mosaic_labels[:, 2] - mosaic_labels[:, 0]
        labels_h = mosaic_labels[:, 3] - mosaic_labels[:, 1]
        mosaic_labels = mosaic_labels[np.logical_and(labels_w > 3, labels_h > 3)]
        return mosaic_img, mosaic_labels, mosaic_mask

    def random_affine(
        self, 
        img,
        mask,
        targets=(),
        target_size=(640, 640),
        mask_scale=4,
        degrees=10,
        translate=0.1,
        scales=0.1,
        shear=10,
    ):
    
        mask_size = (target_size[0] // mask_scale, target_size[1] // mask_scale)
        M, M_mask, scale = self.get_affine_matrix(target_size, mask_size, degrees, translate, scales, shear)

        img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, M_mask, dsize=mask_size, borderValue=(0, 0))

        # Transform label coordinates
        if len(targets) > 0:
            targets = self.apply_affine_to_bboxes(targets, target_size, M, scale)

        return img, mask, targets

    def get_affine_matrix(self, target_size, mask_size, degrees=10, translate=0.1, scales=0.1, shear=10):

        # random value
        def get_aug_params(value, center=0):
            if isinstance(value, float):
                return random.uniform(center - value, center + value)
            elif len(value) == 2:
                return random.uniform(value[0], value[1])
            else:
                raise ValueError(
                    "Affine params should be either a sequence containing two values\
                    or single float values. Got {}".format(value)
                )

        twidth, theight = target_size
        mwidth, mheight = mask_size

        # Rotation and Scale
        angle = get_aug_params(degrees)
        scale = get_aug_params(scales, center=1.0)

        if scale <= 0.0:
            raise ValueError("Argument scale should be positive")

        R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        M = np.ones([2, 3])
        # Shear
        shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
        shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

        M[0] = R[0] + shear_y * R[1]
        M[1] = R[1] + shear_x * R[0]

        # Translation
        translation_x = get_aug_params(translate)   # x translation (pixels)
        translation_y = get_aug_params(translate)   # y translation (pixels)

        target_tx = translation_x * twidth
        target_ty = translation_y * theight
        mask_tx = translation_x * mwidth
        mask_ty = translation_y * mheight

        M[0, 2] = target_tx
        M[1, 2] = target_ty

        M_mask = M.copy()
        M_mask[0, 2] = mask_tx
        M_mask[1, 2] = mask_ty

        return M, M_mask, scale

    def apply_affine_to_bboxes(self, targets, target_size, M, scale):
        num_gts = len(targets)

        # warp corner points
        twidth, theight = target_size
        corner_points = np.ones((4 * num_gts, 3))
        corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            4 * num_gts, 2
        )  # x1y1, x2y2, x1y2, x2y1
        corner_points = corner_points @ M.T  # apply affine transform
        corner_points = corner_points.reshape(num_gts, 8)

        # create new boxes
        corner_xs = corner_points[:, 0::2]
        corner_ys = corner_points[:, 1::2]
        new_bboxes = (
            np.concatenate(
                (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
            )
            .reshape(4, num_gts)
            .T
        )

        # clip boxes
        new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
        new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

        targets[:, :4] = new_bboxes

        return targets
            
    def get_mosaic_coordinate(self, mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
        # index0 to top left part of image
        if mosaic_index == 0:
            x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
            small_coord = w - (x2 - x1), h - (y2 - y1), w, h
        # index1 to top right part of image
        elif mosaic_index == 1:
            x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
            small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
        # index2 to bottom left part of image
        elif mosaic_index == 2:
            x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
            small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
        # index2 to bottom right part of image
        elif mosaic_index == 3:
            x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
            small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
        return (x1, y1, x2, y2), small_coord

    def _mask_generate(self, bboxes, points, img_size, scale=1):
        mask_h = int(img_size[0] / scale)
        mask_w = int(img_size[1] / scale)
        mask_size = (mask_h, mask_w)
        mask = np.zeros(shape=mask_size, dtype=np.float32)
        if len(points) == 0:
            return np.zeros(shape=(mask_h, mask_w, 2), dtype=np.float32)
        points = points / scale
        points = np.concatenate([points[:, :2], points[:, 2:]], axis=0).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], a_min=0, a_max=mask_w-1)
        points[:, 1] = np.clip(points[:, 1], a_min=0, a_max=mask_h-1)
        mask[points[:, 1], points[:, 0]] = 1.0
        # gaussian filter
        gauss_kernel = (11, 11) # must be odd
        mask_blur = cv2.GaussianBlur(mask, gauss_kernel, 1)
        mask_blur = 255 * mask_blur / mask_blur.max()

        # area for loss compute
        mask_pos = np.zeros(shape=mask_size, dtype=np.float32)
        bboxes = (bboxes / scale).astype(np.int32)
        for box in bboxes:
            pad_ = 5
            x1, y1, x2, y2, _ = box
            x_min = max(0, x1 - pad_)
            y_min = max(0, y1 - pad_)
            x_max = min(mask_w, x2 + pad_)
            y_max = min(mask_h, y2 + pad_)
            mask_pos[y_min:y_max, x_min:x_max] = 1.0

        mask = np.stack([mask_blur, mask_pos], axis=2)
        return mask

    def load_image(self, img_path):
        im = cv2.imread(img_path)  # BGR
        assert im is not None, f'Image Not Found {img_path}'
        h0, w0 = im.shape[:2]  # orig hw
        img_size = self.img_size[0]
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def _norm_bboxes(self, bboxes, old_size, new_size):
        bboxes[::2] *= new_size[1] / old_size[1]
        bboxes[1::2] *= new_size[0] / old_size[0]
        return bboxes

    def visual_data_sample(self, idx=None, show_or_save="save"):
        if idx is None:
            idx = random.randint(0, len(self.annotations))
        img, mask, bboxes = self.__getitem__(idx)

        print(img.shape)
        print(mask.shape)
        print(bboxes.shape)
        print(bboxes)
        # draw bboxes
        for bbox in bboxes:
            pt1, pt2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img, pt1, pt2, color=[0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        # for point in points:
        #     pt1, pt2 = (int(point[0]), int(point[1])), (int(point[2]), int(point[3]))
        #     cv2.circle(img, pt1, radius=3, color=[0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        #     cv2.circle(img, pt2, radius=3, color=[0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        # mixup img and mask
        h, w, _ = img.shape
        rmask = cv2.resize(mask[:, :, 0], dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        cmap = plt.get_cmap("gray")
        rgba_img = cmap(rmask / 255)
        rgb_img = np.delete(rgba_img, 3, 2) * 255

        mixup_img = (img[:, :, :] * 0.5 + rgb_img[:, :, ::-1] * 0.5).astype(np.uint8)

        cv2.imwrite("visual_data_sample.png", mixup_img)
        cv2.imwrite("mask.png", mask[:, :, 0])
            
if __name__ == "__main__":
    xml_file = "/home/zhognli/YOLOX/datasets/spindle/train_set/annotations.xml"
    img_dir  = "/home/zhognli/YOLOX/datasets/spindle/train_set/images"
    dataset = CVATVideoDataset(img_dir=img_dir, anno_path=xml_file)
    for idx in range(len(dataset)):
        dataset.visual_data_sample(idx)

    # # dataset self checking
    # for idx in range(len(dataset)):
    #     img, mask, bbox, points = dataset[idx]
    #     print(idx)
    #     print("{}_{}".format(img.shape, mask.shape))