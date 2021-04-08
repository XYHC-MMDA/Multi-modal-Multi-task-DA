# usage: python tmp/visual_point_cloud.py --config config_file_path

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
import os.path as osp
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='train config file path')
args = parser.parse_args()
np.random.seed(0)  # fix dataset items

cfg = Config.fromfile(args.config)
print('cfg loaded')
dataset = build_dataset(cfg.data.train)
print('dataset loaded')
print()

import open3d as o3d
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector

k = len(dataset)
print('len dataset:', k) # 15695
for i in range(5):
    # data_info = dataset.get_data_info(i)
    # print(data_info['pts_filename'])
    # print(data_info['img_filename'][0])

    data = dataset[i]
    seg_points = data['seg_points'].data[:, :3]
    pcd.points = v3d(seg_points)  # seg_points=(N, 3)
    o3d.io.write_point_cloud(osp.join('tmp/mmda_train_usa', f'p4_seg_{i}.pcd'), pcd)
    print('b:', i, len(seg_points))

    points = data['points'].data[:, :3]
    pcd.points = v3d(points)
    o3d.io.write_point_cloud(osp.join('tmp/mmda_train_usa', f'p4_pts_{i}.pcd'), pcd)
    print('b:', i, len(seg_points))

