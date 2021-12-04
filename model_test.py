import torch
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, UnetHead

in_channels = [256, 512, 1024]

backbone =YOLOPAFPN(depth=1.00, 
                    width=1.00, 
                    # in_features=("dark3", "dark4", "dark5"),
                    in_features=("stem", "dark2", "dark3", "dark4", "dark5"),
                    in_channels=[64, 128, 256, 512, 1024],
                    # in_channels=[256, 512, 1024],
                    depthwise=False,
                    act="silu")

det_head = YOLOXHead(num_classes=80, 
                width=1, 
                strides=[8, 16, 32],
                in_channels=[256, 512, 1024],
                act="silu",
                depthwise=False)

mask_head = UnetHead(num_class=1,
                    width=1,
                    strides=[2, 4, 8, 16, 32],
                    in_channels=[64, 128, 256, 512, 1024])

# create the inputs

inps = torch.randn(1, 3, 640, 640)

feature_map = backbone(inps)
print(feature_map)