# usage: python dedebug/calc_class_weights.py --config config_file_path

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
source_train = build_dataset(cfg.data.source_train)
source_test = build_dataset(cfg.data.source_test)
target_train = build_dataset(cfg.data.target_train)
target_test = build_dataset(cfg.data.target_test)
target_val = build_dataset(cfg.data.target_val)
dss = [source_train, source_test, target_train, target_test, target_val]
print('dataset splits loaded')
print()

# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# v3d = o3d.utility.Vector3dVector

num_classes = len(source_train.SEG_CLASSES)
pts_per_class = np.zeros(num_classes, dtype=np.int64)
for ds in dss[:2]:
    k = len(ds)
    print('len dataset:', k)  # 15695
    for i in range(k):
        # data_info = dataset.get_data_info(i)
        # print(data_info['pts_filename'])
        # print(data_info['img_filename'][0])

        data = ds[i]
        # seg_points = data['seg_points'].data[:, :3]
        seg_label = data['seg_label'].data
        import pdb
        pdb.set_trace()
        pts_per_class += np.bincount(seg_label, minlength=num_classes)
        pdb.set_trace()

class_weights = np.log(5 * pts_per_class.sum() / pts_per_class)
print('log smoothed class weights: ', class_weights / class_weights.min())
