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

from collections import OrderedDict
import torch.distributed as dist


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model, dataset, cfg,
                   distributed=False, validate=False, timestamp=None, meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        assert False
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

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

    # discriminators
    seg_discriminator = build_discriminator(cfg.seg_discriminator)
    det_discriminator = build_discriminator(cfg.det_discriminator)
    seg_optimizer = build_optimizer(seg_discriminator, cfg.seg_optimizer)
    det_optimizer = build_optimizer(det_discriminator, cfg.det_optimizer)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = DiscRunner(
        model,
        seg_discriminator,
        det_discriminator,
        seg_optimizer,
        det_optimizer,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting; no optimizer_config
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(
    #         **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # elif distributed and 'type' not in cfg.optimizer_config:
    #     optimizer_config = OptimizerHook(**cfg.optimizer_config)
    # else:
    #     optimizer_config = cfg.optimizer_config

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
                       distributed=False, validate=False, timestamp=None, meta=None):

    logger = get_root_logger(cfg.log_level)
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        assert False
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

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

    # fp16 setting; no optimizer_config
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     optimizer_config = Fp16OptimizerHook(
    #         **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    # elif distributed and 'type' not in cfg.optimizer_config:
    #     optimizer_config = OptimizerHook(**cfg.optimizer_config)
    # else:
    #     optimizer_config = cfg.optimizer_config

    # register hooks; no opimizer_config & momentum_config
    runner.register_training_hooks(cfg.lr_config, checkpoint_config=cfg.checkpoint_config, log_config=cfg.log_config)

    # if distributed:
    #     runner.register_hook(DistSamplerSeedHook())
    # if cfg.resume_from:
    #     runner.resume(cfg.resume_from)
    # elif cfg.load_from:
    #     runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
