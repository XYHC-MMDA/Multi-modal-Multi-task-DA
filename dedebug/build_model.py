# usage: python tmp/visual_point_cloud.py --config config_file_path

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet3d.models import builder
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
import os.path as osp
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('config', help='train config file path')
args = parser.parse_args()
np.random.seed(0)  # fix dataset items

cfg = Config.fromfile(args.config)
print('cfg loaded')
model = builder.build_detector(cfg.model)
print(model.img_backbone)
# f = open('./dedebug/xmuda_UnetResNet34.txt', 'w')
# f.write(str(model.img_backbone))
# f.close()

# dataset = build_dataset(cfg.data.source_train)
# print('dataset loaded')

