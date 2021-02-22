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


def points_in_box(corners, points):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param corners: (3, 8).
    :param points: <torch.float: 3, n>.
    :return: <torch.bool: n, >.
    """

    p1 = corners[:, 0]
    p_x = corners[:, 3]
    p_y = corners[:, 4]
    p_z = corners[:, 1]
    # p_x = corners[:, 4]
    # p_y = corners[:, 1]
    # p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = torch.matmul(i, v)
    jv = torch.matmul(j, v)
    kv = torch.matmul(k, v)

    mask_x = torch.logical_and(0 <= iv, iv <= torch.matmul(i, i))
    mask_y = torch.logical_and(0 <= jv, jv <= torch.matmul(j, j))
    mask_z = torch.logical_and(0 <= kv, kv <= torch.matmul(k, k))
    mask = torch.logical_and(torch.logical_and(mask_x, mask_y), mask_z)

    return mask

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
dataset = build_dataset(cfg.data.train)
print('Dataset Loaded.')
num_classes = 5

for idx in range(len(dataset)):
    data = dataset[idx]
    points = data['points'].data[:, :3]
    seg_points = data['seg_points'].data[:, :3]  # (N, 4) -> (N, 3)
    seg_labels = data['seg_label'].data  # (N, ) 
    gt_bboxes_3d = data['gt_bboxes_3d'].data  # tensor=(M, 9)
    gt_labels_3d = data['gt_labels_3d'].data  # (M, )


    # LiDARInstance3DBoxes points_in_boxes
    tensor_boxes = gt_bboxes_3d.tensor.cuda()
    pts1 = seg_points.cuda()
    start = time.time()
    tensor_boxes = tensor_boxes[:, :7]
    box_idx = points_in_boxes_gpu(pts1.unsqueeze(0), tensor_boxes.unsqueeze(0)).squeeze(0)
    # boxes = LiDARInstance3DBoxes(gt_bboxes_3d.tensor[:, :7])
    # box_idx = boxes.points_in_boxes(seg_points[:, :3].cuda())

    # 1.
    # fake_labels = torch.tensor([num_classes-1] * len(seg_labels))
    # for i in range(len(box_idx)):
    #     if box_idx[i] != -1:
    #         fake_labels[i] = gt_labels_3d[box_idx[i]]

    # 2.
    # fake_labels = torch.tensor(list(map(lambda x: gt_labels_3d[x] if x >= 0 else num_classes-1, box_idx)))

    # 3.
    fake_labels = torch.tensor([num_classes-1] * len(seg_labels))
    mask = box_idx != -1
    fake_labels[mask] = torch.tensor(list(map(lambda x: gt_labels_3d[x], box_idx[mask])), dtype=torch.long)
    end = time.time()
    t1 = end - start

    evaluator = SegEvaluator(class_names=dataset.SEG_CLASSES)
    evaluator.update(fake_labels.numpy(), seg_labels.numpy())
    print(evaluator.print_table())
    print('overall_acc:', evaluator.overall_acc)
    print('overall_iou:', evaluator.overall_iou)

    # nuscenes points_in_box
    start = time.time()
    allcorners = gt_bboxes_3d.corners  # (N, 8, 3)
    fake_labels = torch.tensor([num_classes-1] * len(seg_labels))
    for i, (corners, label) in enumerate(zip(allcorners, gt_labels_3d)):
        mask = points_in_box(corners.T, seg_points[:, :3].T)
        fake_labels[mask] = label
    end = time.time()
    t2 = end - start

    evaluator = SegEvaluator(class_names=dataset.SEG_CLASSES)
    evaluator.update(fake_labels.numpy(), seg_labels.numpy())
    print(evaluator.print_table())
    print('overall_acc:', evaluator.overall_acc)
    print('overall_iou:', evaluator.overall_iou)

    print('t1: %.4fs t2: %.4fs' % (t1, t2))
    pdb.set_trace()
