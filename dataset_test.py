from yolox.data.datasets.microtube import VOCDetSegDataset
from yolox.data import TrainTransform
import cv2
import numpy as np

train_img_dir = "datasets/cvat_example/task_test-2021_11_23_14_42_29-cvat for images 1.1/images"
train_anno_path = "datasets/cvat_example/task_test-2021_11_23_14_42_29-cvat for images 1.1/annotations.xml"
dataset = VOCDetSegDataset(img_dir=train_img_dir, 
                        anno_path=train_anno_path,
                        img_size=(640, 640),
                        preproc=TrainTransform(
                            max_labels=50,
                            flip_prob=0,
                            hsv_prob=0)
                        )
image_t, mask_t, padded_labels, points = dataset[0]

# convert 
img = np.ascontiguousarray(np.transpose(image_t, axes=(1, 2, 0)))
cv2.imwrite("img_enhanced.png", img)

mask = mask_t

for box in padded_labels:
    if not np.all(box==0):
        label, xc, yc, w, h = box
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, [255, 0, 0], thickness=1, lineType=cv2.LINE_AA)
cv2.imwrite("labeld_img.png", img)
cv2.imwrite("mask.png", mask_t)