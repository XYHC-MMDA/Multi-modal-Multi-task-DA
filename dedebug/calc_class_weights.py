# usage: python dedebug/calc_class_weights.py --config config_file_path
# usa-sng: [2.15420616, 3.2978224, 4.44739869, 2.85514612, 1., 2.84083026, 2.15164313, 2.75812384, 1.54123938, 1.84512741, 2.25738817]
# usa-sng: [2.154, 3.298, 4.447, 2.855, 1., 2.841, 2.152, 2.758, 1.541, 1.845, 2.257]

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
# target_train = build_dataset(cfg.data.target_train)
# target_test = build_dataset(cfg.data.target_test)
# target_val = build_dataset(cfg.data.target_val)
dss = [source_train, source_test]
print('dataset splits loaded')
print()

# import open3d as o3d
# pcd = o3d.geometry.PointCloud()
# v3d = o3d.utility.Vector3dVector

num_classes = len(source_train.SEG_CLASSES)
source_train_count = np.zeros(num_classes, dtype=np.int64)
source_test_count = np.zeros(num_classes, dtype=np.int64)

k = len(source_train)
print('source_train:', k)  # 15695
shuffle = True
if shuffle:
    index = np.random.permutation(k)
else:
    index = np.arange(k)
for i in range(k):
    data = source_train[index[i]]
    # seg_points = data['seg_points'].data[:, :3]
    seg_label = data['seg_label'].data
    count = np.bincount(seg_label, minlength=num_classes)
    print(count)
    import pdb
    pdb.set_trace()
    source_train_count += count
    if i % 100 == 99:
        print(f'[{i}] -', source_train_count)
    
k = len(source_test)
print('source_test:', k)  # 15695
for i in range(k):
    data = source_test[i]
    seg_label = data['seg_label'].data
    count = np.bincount(seg_label, minlength=num_classes)
    source_test_count += count
    if i % 100 == 99:
        print(f'[{i}] -', source_test_count)

print('source_train_count:', source_train_count)
print('source_test_count:', source_test_count)
pts_per_class = source_train_count + source_test_count
print('total:', pts_per_class)
class_weights = np.log(5 * pts_per_class.sum() / pts_per_class)
print('log smoothed class weights: ', class_weights / class_weights.min())
