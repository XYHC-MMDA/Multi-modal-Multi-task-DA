from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
import argparse
import numpy as np
import torch
from PIL import Image
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
dataset = build_dataset(cfg.data.train)
print('Dataset Loaded.')

for idx in range(len(dataset)):
    data = dataset[idx]
    seg_points = data['seg_points'].data
    seg_labels = data['seg_label'].data
    gt_bboxes_3d = data['gt_bboxes_3d'].data
    gt_labels_3d = data['gt_labels_3d'].data
    import pdb
    pdb.set_trace()
