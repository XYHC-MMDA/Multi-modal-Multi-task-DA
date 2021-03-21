import argparse
import time
import mmcv
import os
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.apis import seg_test_with_loss
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
from mmdet.apis import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='segmentation test a model on source/target val/test split')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--split', type=str, help='source_test, target_test, target_val')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg_start_time = time.time()
    cfg = Config.fromfile(args.config)
    cfg_last = time.time() - cfg_start_time
    print('cfg time:', cfg_last)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # cfg.data.test.test_mode = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    dataset_start_time = time.time()
    # dataset = build_dataset(cfg.data.test)
    dataset = build_dataset(cfg.data.get(args.split))
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
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    model_time = time.time() - model_start_time
    print('model time:', model_time)

    model = MMDataParallel(model, device_ids=[0])
    seg_test_with_loss(model, data_loader)


if __name__ == '__main__':
    main()
