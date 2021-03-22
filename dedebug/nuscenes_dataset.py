from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction

import argparse
import numpy as torch
import torch
from PIL import Image
import os
import pdb
from tools.evaluator import SegEvaluator
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
dataset = build_dataset(cfg.data.train)
print('Dataset Loaded.')
num_classes = 5

evaluator = SegEvaluator(class_names=dataset.SEG_CLASSES)
for idx in range(len(dataset)):
    data = dataset[idx]
    points = data['points'].data[:, :3]
    seg_points = data['seg_points'].data[:, :3]  # (N, 4) -> (N, 3)
    seg_labels = data['seg_label'].data  # (N, ) 
    # gt_bboxes_3d = data['gt_bboxes_3d'].data  # tensor=(M, 9)
    # gt_labels_3d = data['gt_labels_3d'].data  # (M, )


    # LiDARInstance3DBoxes points_in_boxes
    tensor_boxes = gt_bboxes_3d.tensor.cuda()
    pts1 = seg_points.cuda()
    start = time.time()
    tensor_boxes = tensor_boxes[:, :7]
    box_idx = points_in_boxes_gpu(pts1.unsqueeze(0), tensor_boxes.unsqueeze(0)).squeeze(0)
    # boxes = LiDARInstance3DBoxes(gt_bboxes_3d.tensor[:, :7])
    # box_idx = boxes.points_in_boxes(seg_points[:, :3].cuda())
    fake_labels = torch.tensor([num_classes-1] * len(seg_labels))
    mask = box_idx != -1
    fake_labels[mask] = torch.tensor(list(map(lambda x: gt_labels_3d[x], box_idx[mask])), dtype=torch.long)
    end = time.time()
    t1 = end - start

    evaluator.update(fake_labels.numpy(), seg_labels.numpy())
    if idx % 100 == 0:
        print(idx)
        print(evaluator.print_table())
        print('overall_acc:', evaluator.overall_acc)
        print('overall_iou:', evaluator.overall_iou)

print(evaluator.print_table())
print('overall_acc:', evaluator.overall_acc)
print('overall_iou:', evaluator.overall_iou)
