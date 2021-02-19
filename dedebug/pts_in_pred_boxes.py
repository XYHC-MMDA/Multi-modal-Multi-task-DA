from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
import argparse
import numpy as torch
import torch
from PIL import Image
import os


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
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = torch.dot(i, v)
    jv = torch.dot(j, v)
    kv = torch.dot(k, v)

    mask_x = torch.logical_and(0 <= iv, iv <= torch.dot(i, i))
    mask_y = torch.logical_and(0 <= jv, jv <= torch.dot(j, j))
    mask_z = torch.logical_and(0 <= kv, kv <= torch.dot(k, k))
    mask = torch.logical_and(torch.logical_and(mask_x, mask_y), mask_z)

    return mask

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
    gt_bboxes_3d = data['gt_bboxes_3d'].data  # tensor=(N, 9)
    gt_labels_3d = data['gt_labels_3d'].data  # (N, )

    allcorners = gt_bboxes_3d.corners  # (N, 8, 3)
    fake_labels = torch.tensor([4] * len(seg_labels))
    for i, (corners, label) in enumerate(zip(allcorners, gt_labels_3d)):
        mask = points_in_box(corners.T, seg_points.T)
        fake_labels[mask] = label


