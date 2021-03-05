import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner, RUNNERS,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger
from mmdet3d.models import build_discriminator
from mmdet3d.runners import RepRunner
from mmdet3d.parallel import MyDataParallel, TCDataParallel


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model, dataset, cfg, distributed=False, timestamp=None, meta=None):
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
    # model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    model = MyDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # discriminators
    seg_disc, seg_opt = None, None
    if cfg.seg_discriminator is not None:
        seg_disc = build_discriminator(cfg.seg_discriminator).cuda()
        seg_opt = build_optimizer(seg_disc, cfg.seg_optimizer)
    det_disc, det_opt = None, None
    if cfg.det_discriminator is not None:
        det_disc = build_discriminator(cfg.det_discriminator).cuda()
        det_opt = build_optimizer(det_disc, cfg.det_optimizer)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    PRunner = RUNNERS.get(cfg.runner)
    runner_kwargs = dict()
    for key in ['lambda_GANLoss', 'src_acc_threshold', 'tgt_acc_threshold']:
        if not hasattr(cfg, key):
            continue
        runner_kwargs[key] = getattr(cfg, key)
    runner = PRunner(model, seg_disc=seg_disc, seg_opt=seg_opt, det_disc=det_disc, det_opt=det_opt, **runner_kwargs,
                     optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)
    runner.timestamp = timestamp

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_single_seg_detector(model, dataset, cfg, distributed=False, timestamp=None, meta=None):
    # dataset: [src_dataset, tgt_dataset]
    logger = get_root_logger(cfg.log_level)
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
    # model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    model = MyDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    runner_kwargs = dict()
    for disc_key, opt_key in [('seg_discriminator', 'seg_optimizer'),
                              ('img_disc', 'img_opt'),
                              ('lidar_disc', 'lidar_opt')]:
        if not hasattr(cfg, disc_key):
            continue
        disc = build_discriminator(getattr(cfg, disc_key)).cuda()
        opt = build_optimizer(disc, getattr(cfg, opt_key))
        if disc_key == 'seg_discriminator':
            disc_key, opt_key = 'seg_disc', 'seg_opt'
        runner_kwargs[disc_key] = disc
        runner_kwargs[opt_key] = opt

    optimizer = build_optimizer(model, cfg.optimizer)
    PRunner = RUNNERS.get(cfg.runner)
    for key in ['lambda_GANLoss', 'src_acc_threshold', 'tgt_acc_threshold', 'return_fusion_feats',
                'lambda_img', 'lambda_lidar']:
        if not hasattr(cfg, key):
            continue
        runner_kwargs[key] = getattr(cfg, key)
    runner = PRunner(model, **runner_kwargs,
                     optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)
    # runner = PRunner(model, seg_disc=seg_disc, seg_opt=seg_opt, lambda_GANLoss=cfg.lambda_GANLoss,
    #                  return_fusion_feats=cfg.return_fusion_feats, optimizer=optimizer, work_dir=cfg.work_dir,
    #                  logger=logger, meta=meta)
    runner.timestamp = timestamp

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
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
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_tc_detector(model, dataset, cfg, distributed=False, timestamp=None, meta=None):
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
        model = TCDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    PRunner = RUNNERS.get(cfg.runner)
    runner = PRunner(model, cfg=cfg, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_general_detector(model, dataset, cfg, distributed=False, timestamp=None, meta=None):
    # dataset: [src_dataset, tgt_dataset]
    logger = get_root_logger(cfg.log_level)
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
    model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    runner_kwargs = dict()
    for disc_key, opt_key in [('discriminator', 'disc_optimizer'),
                              ('lidar_disc', 'lidar_opt')]:
        if not hasattr(cfg, disc_key):
            continue
        disc = build_discriminator(getattr(cfg, disc_key)).cuda()
        opt = build_optimizer(disc, getattr(cfg, opt_key))
        runner_kwargs[disc_key] = disc
        runner_kwargs[opt_key] = opt

    optimizer = build_optimizer(model, cfg.optimizer)
    PRunner = RUNNERS.get(cfg.runner)
    for key in ['lambda_GANLoss', 'src_acc_threshold', 'tgt_acc_threshold', 'return_fusion_feats',
                'lambda_img', 'lambda_lidar']:
        if not hasattr(cfg, key):
            continue
        runner_kwargs[key] = getattr(cfg, key)
    runner = PRunner(model, **runner_kwargs,
                     optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta)
    runner.timestamp = timestamp

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)

