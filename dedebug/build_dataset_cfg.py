# usage: python tmp/visual_point_cloud.py --config config_file_path

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
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
# dataset = build_dataset(cfg.data.source_train)
dataset = build_dataset(cfg.data.source_test)
print('dataset loaded')
print()

# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# v3d = o3d.utility.Vector3dVector

k = len(dataset)
print('len dataset:', k) # 15695
for i in range(5):
    # data_info = dataset.get_data_info(i)
    # print(data_info['pts_filename'])
    # print(data_info['img_filename'][0])

    data = dataset[i]
    seg_points = data['seg_points'].data[:, :3]
    si = data['seg_pts_indices'].data
    scn_coords = data['scn_coords']
    seg_label = data['seg_label']
    s = []
    c = []
    cnt = 0
    for i, x in enumerate(scn_coords.data):
        x = list(x.numpy())
        if x not in s:
            s.append(x)
            c.append(i)
        else:
            cnt += 1
            print(c[s.index(x)], x)
    import pdb
    pdb.set_trace()
    # print('b:', i, len(seg_points))

