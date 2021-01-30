import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
from mmdet3d.models import build_discriminator
from mmdet3d.runners import DiscRunner, RepRunner
from mmdet3d.parallel import MyDataParallel


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model, dataset, cfg,
                   distributed=False, timestamp=None, meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    # dataset: [src_dataset, tgt_dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    # model = MyDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # discriminators
    seg_discriminator = build_discriminator(cfg.seg_discriminator)
    det_discriminator = build_discriminator(cfg.det_discriminator)
    seg_optimizer = build_optimizer(seg_discriminator, cfg.seg_optimizer)
    det_optimizer = build_optimizer(det_discriminator, cfg.det_optimizer)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = DiscRunner(model, seg_discriminator, det_discriminator, seg_optimizer, det_optimizer,
                        optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)
    runner.timestamp = timestamp

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())
    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def rep_train_detector(model, dataset, cfg,
                       distributed=False, timestamp=None, meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        assert False
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = RepRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())
    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
