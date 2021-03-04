import mmcv
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
from mmdet.datasets import build_dataloader
from mmcv import Config, DictAction
from mmdet3d.models import build_detector
from mmdet3d.apis import set_random_seed
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

import argparse
import numpy as torch
import torch
from PIL import Image
import os
import pdb
from tools.evaluator import SegEvaluator
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_gpu
import time

parser = argparse.ArgumentParser()
parser.add_argument('config', help='test config file path')
parser.add_argument('checkpoint', help='checkpoint file')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

cfg = Config.fromfile(args.config)
dataset = build_dataset(cfg.data.train)
print('Dataset Loaded.')
num_classes = 5


def calc(model, data_loader):
    model.eval()
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    seg_eval = SegEvaluator(class_names=dataset.SEG_CLASSES)
    det_eval = SegEvaluator(class_names=dataset.SEG_CLASSES)
    print('\nStart Test Loop')
    # print('batch_size:', data_loader.batch_size)  # 1
    for idx, data in enumerate(data_loader):
        # print(type(data['img'][0]))  # DataContainter
        with torch.no_grad():
            seg_res, box_res = model(return_loss=False, rescale=True, **data)
        # len(box_res) == batch_size == 1
        # box_res: [dict('pts_bbox'=dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels))]

        # handle seg
        seg_label = data['seg_label'][0].data[0]  # list of tensor
        seg_pts_indices = data['seg_pts_indices'][0].data[0]  # list of tensor
        seg_points = data['seg_points'][0].data[0]
        seg_pred = seg_res.argmax(1).cpu().numpy()
        pred_list = []
        gt_list = []
        left_idx = 0
        for i in range(len(seg_label)):
            # num_points = len(seg_pts_indices[i])
            assert len(seg_label[i]) == len(seg_pts_indices[i])
            num_points = len(seg_label[i])
            right_idx = left_idx + num_points
            pred_list.append(seg_pred[left_idx: right_idx])
            gt_list.append(seg_label[i].numpy())
            left_idx = right_idx
        seg_eval.batch_update(pred_list, gt_list)

        # handle det
        dic = box_res[0]['pts_bbox']
        tensor_boxes = dic['boxes_3d'].tensor[:, :7].cuda()
        labels = dic['labels_3d']

        num_seg_pts = len(seg_points[0])
        num_pred_boxes = len(tensor_boxes)
        fake_labels = torch.tensor([4] * num_seg_pts)
        box_idx = points_in_boxes_gpu(seg_points[0].cuda().unsqueeze(0), tensor_boxes.unsqueeze(0)).squeeze(0)
        for i in range(num_pred_boxes):
            mask = box_idx == i  # select points in i_th box
            fake_labels[mask] = labels[i]
        det_eval.update(fake_labels.numpy(), seg_label[0])


        # progress bar
        batch_size = len(box_res)
        for _ in range(batch_size):
            prog_bar.update()

    print(seg_eval.print_table())
    print('overall_acc:', seg_eval.overall_acc)
    print('overall_iou:', seg_eval.overall_iou)
    print(det_eval.print_table())
    print('overall_acc:', det_eval.overall_acc)
    print('overall_iou:', det_eval.overall_iou)


if __name__ == '__main__':
    cfg_start_time = time.time()
    cfg = Config.fromfile(args.config)
    cfg_last = time.time() - cfg_start_time
    print('cfg time:', cfg_last)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    dataset_start_time = time.time()
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    dataset_last = time.time() - dataset_start_time
    print('dataset & dataloader time:', dataset_last)

    # build the model and load checkpoint
    model_start_time = time.time()
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model_time = time.time() - model_start_time
    print('model time:', model_time)

    model = MMDataParallel(model, device_ids=[0])
    calc(model, data_loader)

