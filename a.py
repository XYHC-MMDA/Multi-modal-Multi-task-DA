from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
import argparse
import numpy as np
import torch
from PIL import Image
import os

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--config', help='train config file path')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
print('cfg loaded')
dataset = build_dataset(cfg.data.train)
print('dataset loaded')
print()

maxp=0
minp=1000000
print(len(dataset))
f = open('a.txt','w')
for idx in range(len(dataset)):
    data = dataset[idx]
    tmp = len(data['seg_points'].data)
    # print(data['seg_points'].data.shape) # torch.Size([2xxx, 4])
    maxp=max(maxp, tmp)
    minp=min(minp, tmp)
    f.write(f'{idx} {tmp}\n')
    if idx % 100 == 0:
        print(idx, minp, maxp)
        f.flush()
print('result')
print(minp, maxp)
f.write('result\n')
f.write(f'{minp} {maxp}\n')
exit(0)
    
print_dataset = True 
data_idx = 1
if print_dataset:
    data_info = dataset.get_data_info(data_idx)
    print('get_data_info:')
    print(data_info.keys())
    # 'pts_filename', 'img_filename', 'seglabel_filename'
    img_filepath = data_info['img_filename'][0]
    pts_filepath = data_info['pts_filename']
    # num_points is multiple of 32(34720, 34688, 34752..)
    print(img_filepath)
    print(pts_filepath)
    print()
exit(0)

if print_dataset:
    data = dataset[data_idx]
    print('getitem:')
    print(data.keys())
    print()
    exit(0)
    gt_bboxes = data['gt_bboxes_3d'].data
    print(gt_bboxes.center.shape)
    print(gt_bboxes.center)
    minx, maxx = 0, 0
    mind, maxd = 1000, -1000
    for i in range(len(dataset)):
        if i % 20 == 0:
            print(f'[{i}]', minx, maxx)
        pts = dataset[i]['points'].data.numpy()
        # degrees = np.arctan2(pts[:, 1] , pts[:, 0]) / np.pi * 180
        # mind = min(mind, np.min(degrees))
        # maxd = max(maxd, np.max(degrees))
        minx = min(minx, np.min(pts[:, 0]))
        maxx = max(maxx, np.max(pts[:, 0]))
    exit(0)
    # print(pts_seg.shape)
    # print(torch.min(pts_seg[:, 0]), torch.max(pts_seg[:, 0]))
    # print(torch.min(pts_seg[:, 1]), torch.max(pts_seg[:, 1]))
    # print(torch.min(pts_seg[:, 2]), torch.max(pts_seg[:, 2]))
    # degrees = np.arctan2(pts_seg[:, 2] , pts_seg[:, 0]) / np.pi * 180
    # print(np.max(degrees), np.min(degrees))

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
import open3d as o3d
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector

pts_dataset = data['points'].data
pcd.points = v3d(pts_dataset[:, :3].numpy())
o3d.io.write_point_cloud(f'debug/train/{data_idx}_pts_det_cam.pcd', pcd)

pcd.points = v3d(np.fromfile(pts_filepath, dtype=np.float32).reshape(-1, 5)[:, :3])
o3d.io.write_point_cloud(f'debug/train/{data_idx}_pts_all.pcd', pcd)
exit(0)

img = Image.open(img_filepath)
img.save(f'debug/mini_train/{data_idx}_front.png')

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
exit(0)

from tools.draw_utils import draw_box3d_image
box_img = draw_box3d_image(img, corner_img_indices)
box_img = Image.fromarray(box_img)
box_img.save(f'debug/mini_train/{data_idx}_front_box.png')
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
