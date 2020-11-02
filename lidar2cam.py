from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmcv import Config, DictAction
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('--config', help='train config file path')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
cfg.data.train.modality.use_camera=True
print('cfg loaded')

dataset = build_dataset(cfg.data.train)
print('dataset loaded')
data = dataset.get_data_info(0)
print(data['img_filename'][0])
# for k, v in data.items():
#   print(k, type(v))
pts_lidar = np.fromfile(data['pts_filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
N = pts_lidar.shape[0]
print('pts total:', N)
# pts = pts[:, :3]
rot = data['lidar2img'][0]
pts = np.concatenate([pts_lidar, np.ones((N, 1))], axis=1) @ rot.T
pts = pts[:, :3]

mask = np.ones(N, dtype=bool)
mask = np.logical_and(mask, pts[: ,2] > 1e-5)
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
pcd = o3d.geometry.PointCloud()
v3d = o3d.utility.Vector3dVector
pcd.points = v3d(pts_lidar)
o3d.io.write_point_cloud('sample.pcd', pcd)


