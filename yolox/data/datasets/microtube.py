import os
import os.path
import pickle
import xml.etree.ElementTree as ET
from loguru import logger

import cv2
import numpy as np
import random
from yolox.data.datasets.datasets_wrapper import Dataset

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import Dataset
# from .voc_classes import VOC_CLASSES

class VOCDetSegDataset(Dataset):
    """
    VOC Detection Dataset Object

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
        # target_transform=AnnotationTransform()
    ):
        super().__init__(img_size)
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.img_size = img_size    # (img_h, img_w)
        self.preproc = preproc
        
        # load the annotation data
        self.annotations = self._load_annotations()
        
    def _load_annotations(self):
        """parser the xml annotation file"""
        # step1: load the xml file
        xml_root = ET.parse(self.anno_path).getroot()
        anno_list = []
        img_lost  = []
        for children in xml_root.iter("image"):
            print(children.tag, ":", children.attrib)
            img_attrib = children.attrib
            img_name = img_attrib["name"]
            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                img_attrib["img_path"] = img_path
            else:
                img_lost.append(img_path)
                continue
            img_attrib["bboxes"] = []
            img_attrib["points"] = []
            for box in children.iter("box"):
                box_attrib = box.attrib
                img_attrib["bboxes"].append([
                    int(float(box_attrib["xtl"])),
                    int(float(box_attrib["ytl"])),
                    int(float(box_attrib["xbr"])),
                    int(float(box_attrib["ybr"]))
                ])

            for points in children.iter("points"):
                points_str = points.attrib["points"]
                p1, p2 = points_str.split(";")
                p1 = [int(float(p)) for p in p1.split(",")]
                p2 = [int(float(p)) for p in p2.split(",")]
                img_attrib["points"].append([*p1, *p2])
            anno_list.append(img_attrib)
        return anno_list
    
    def __len__(self):
        return len(self.annotations)

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

        cur_anno = self.annotations[idx]
        
        # load the img
        img, hw_origin, hw_resize = self.load_image(idx)
        
        # load bboxes
        bboxes = np.asarray(cur_anno["bboxes"], dtype=np.float32)
        bboxes = self._norm_bboxes(bboxes, old_size=hw_origin, new_size=hw_resize)
        bboxes = np.concatenate([bboxes, np.zeros(shape=(len(bboxes), 1), dtype=np.float32)], axis=1)

        # load points
        points = np.asarray(cur_anno["points"], dtype=np.float32)
        points = self._norm_bboxes(points, old_size=hw_origin, new_size=hw_resize)
        mask = self._mask_generate(points, hw_resize)
        mask = mask[:, :, np.newaxis]

        if self.preproc is not None:
            # concat the image and mask together
            img, mask, bboxes = self.preproc(img, bboxes, self.input_dim, mask)

        return img, mask, bboxes, points

    def _mask_generate(self, points, img_size):
        mask = np.zeros(shape=img_size, dtype=np.float32)
        points = np.concatenate([points[:, :2], points[:, 2:]], axis=0).astype(np.int32)
        mask[points[:, 1], points[:, 0]] = 1.0
        # gaussian filter
        gauss_kernel = (11, 11) # must be odd
        mask_blur = cv2.GaussianBlur(mask, gauss_kernel, 1.3)
        return 255 * mask_blur / mask_blur.max()

    def load_image(self, idx):
        path = self.annotations[idx]["img_path"]
        im = cv2.imread(path)  # BGR
        assert im is not None, f'Image Not Found {path}'
        h0, w0 = im.shape[:2]  # orig hw
        img_size = self.img_size[0]
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR)
        return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized

    def _norm_bboxes(self, bboxes, old_size, new_size):
        bboxes[::2] *= new_size[1] / old_size[1]
        bboxes[1::2] *= new_size[0] / old_size[0]
        return bboxes

    def visual_data_sample(self, idx=None, show_or_save="save"):
        if idx is None:
            idx = random.randint(0, len(self.annotations))
        img, mask, bboxes, points = self.__getitem__(idx)
        # draw bboxes
        for bbox in bboxes:
            pt1, pt2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(img, pt1, pt2, color=[255, 0, 0], thickness=1, lineType=cv2.LINE_AA)
        for point in points:
            pt1, pt2 = (int(point[0]), int(point[1])), (int(point[2]), int(point[3]))
            cv2.circle(img, pt1, radius=3, color=[0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(img, pt2, radius=3, color=[0, 255, 0], thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite("visual_data_sample.png", img)

        mask = np.int32(mask) 
        cv2.imwrite("mask.png", mask)
        
if __name__ == "__main__":
    xml_file = "datasets/cvat_example/task_test-2021_11_23_14_42_29-cvat for images 1.1/annotations.xml"
    img_dir  = "/home/zhognli/YOLOX/datasets/cvat_example/task_test-2021_11_23_14_42_29-cvat for images 1.1/images"
    dataset = VOCDetSegDataset(img_dir=img_dir, anno_path=xml_file,)
    img, mask, bbox, points = dataset[0]
    print(img.shape)
    print(mask.shape)
    print(bbox.shape)
    print(points.shape)
    dataset.visual_data_sample(0)