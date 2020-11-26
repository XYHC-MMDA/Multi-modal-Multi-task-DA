from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
import argparse
import numpy as np
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--config', help='train config file path')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
print('cfg loaded')
dataset = build_dataset(cfg.data.mini_train)
print('dataset loaded')
print()

print_dataset = True 
data_idx = 91
if print_dataset:
    data_info = dataset.get_data_info(data_idx)
    print('get_data_info:')
    print(data_info.keys())
    img_filepath = data_info['img_filename'][0]
    pts_filepath = data_info['pts_filename']
    print(img_filepath)
    print(pts_filepath)
    print()

if print_dataset:
    data = dataset[data_idx]
    print('getitem:')
    print(data.keys())
    print()
    pts_seg = data['points_seg_cam']
    degrees = np.arctan2(pts_seg[:, 2] , pts_seg[:, 0]) / np.pi * 180
    print(np.max(degrees), np.min(degrees))
    exit(0)

dataloader = build_dataloader(
    dataset,
    cfg.data.samples_per_gpu,
    cfg.data.workers_per_gpu,
    1, # num_gpu
    dist=False,
    shuffle=False
)
print('dataloader finished')
print()

data_batch = iter(dataloader).next()
print('data_batch:', data_batch.keys())
# print(data_batch['gt_bboxes_3d']._data[0][0].center)  # tensor;(N,3) 
# print(len(data_batch['img_indices']._data[0]))  # batch_size
# print(data_batch['img_indices']._data[0][0].shape)  # tensor;(N,4) 
# print(len(data_batch['points']._data[0]))
# print(data_batch['points']._data[0][0].shape)  # tensor; (N, 4)
# print(data_batch['img']._data[0].shape)  #(B, 3, 225, 400)

###########################################
from PIL import Image
img = Image.open(img_filepath)
img = np.array(img)

gt_bboxes_3d = data['gt_bboxes_3d'].data
# gt_bboxes_3d = data_info['ann_info']['gt_bboxes_3d']
# print(np.concatenate([gt_bboxes_3d.center, gt_bboxes_3d.dims], axis=1).astype(np.int64))
rot = data_info['lidar2img'][0]
corners = gt_bboxes_3d.corners  # (N, 8, 3)
corners = corners.reshape(-1, 3)
pts = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1) @ rot.T
pts = pts[:, :3]
pts[:, 0] /= pts[:, 2]
pts[:, 1] /= pts[:, 2]
corner_img_indices = pts.reshape(-1, 8, 3).astype(np.int64)

from tools.draw_utils import draw_box3d_image
box_img = draw_box3d_image(img, corner_img_indices)
box_img = Image.fromarray(box_img)
box_img.save(f'debug/mini_train/{data_idx}_front_box.png')
exit(0)

###########################################
import open3d as o3d
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector

pts_dataset = data['points'].data
pcd.points = v3d(pts_dataset[:, :3].numpy())
o3d.io.write_point_cloud(f'debug/mini_train/{data_idx}_half.pcd', pcd)

pcd.points = v3d(np.fromfile(pts_filepath, dtype=np.float32).reshape(-1, 5)[:, :3])
o3d.io.write_point_cloud(f'debug/mini_train/{data_idx}_all.pcd', pcd)

img = Image.open(img_filepath)
img.save(f'debug/mini_train/{data_idx}_front.png')
exit(0)

###########################################
pts_lidar = np.fromfile(data['pts_filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
N = pts_lidar.shape[0]
print('pts total:', N)
# pts = pts[:, :3]
rot = data['lidar2img'][0]
pts = np.concatenate([pts_lidar, np.ones((N, 1))], axis=1) @ rot.T
pts = pts[:, :3]

mask = np.ones(N, dtype=bool)
mask = np.logical_and(mask, pts[:, 2] > 1e-5)
pts = pts[mask]
pts_lidar = pts_lidar[mask]
pts[:, 0] /= pts[:, 2]
pts[:, 1] /= pts[:, 2]
mask = np.ones(pts.shape[0], dtype=bool)
mask = np.logical_and(mask, 0 < pts[:, 0])
mask = np.logical_and(mask, pts[:, 0] < 1600)
mask = np.logical_and(mask, 0 < pts[:, 1])
mask = np.logical_and(mask, pts[:, 1] < 900)
pts = pts[mask]
pts_lidar = pts_lidar[mask]
print('pts:', pts.shape[0])


import open3d as o3d
from PIL import Image
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector
pcd.points = v3d(pts_lidar)
o3d.io.write_point_cloud('sample.pcd', pcd)
