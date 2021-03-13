import os.path as osp
import platform
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import mmcv
from mmcv.runner import BaseRunner, RUNNERS, save_checkpoint, get_host_info, build_optimizer
from mmdet3d.models.builder import build_loss
from mmdet3d.apis import parse_losses, set_requires_grad
from mmdet3d.models import build_discriminator


@RUNNERS.register_module()
class ContrastRunnerV0(BaseRunner):
    def __init__(self,
                 model,
                 cfg,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(ContrastRunnerV0, self).__init__(model,
                                                 batch_processor,
                                                 optimizer,
                                                 work_dir,
                                                 logger,
                                                 meta,
                                                 max_iters,
                                                 max_epochs)
        self.lambda_contrast = cfg.lambda_contrast
        self.max_pts = cfg.max_pts
        self.contrast_criterion = build_loss(cfg.contrast_criterion)

    def train(self, src_data_loader, tgt_data_loader):
        self.model.train()
        self.mode = 'train'
        self.data_loader = src_data_loader
        tgt_data_iter = iter(tgt_data_loader)
        assert len(tgt_data_loader) <= len(src_data_loader)

        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, src_data_batch in enumerate(self.data_loader):
            # before train iter
            self._inner_iter = i
            self.call_hook('before_train_iter')

            # fetch target data batch
            tgt_data_batch = next(tgt_data_iter, None)
            if tgt_data_batch is None:
                tgt_data_iter = iter(tgt_data_loader)
                tgt_data_batch = next(tgt_data_iter)

            # ------------------------
            # forward source & target
            # ------------------------
            losses, src_img_feats, src_pts_feats, src_pts_indices = self.model(**src_data_batch)
            tgt_img_feats, tgt_pts_feats, tgt_pts_indices = self.model(**tgt_data_batch, target=True)
            # img_feats: (N, C, H, W)=(4, 64, 225, 400); pts_feats: list of (N, C); pts_indices: list of (N, 2)

            # ------------------------
            # consistency loss
            # ------------------------
            # set_requires_grad([self.disc], requires_grad=True)

            # source consistency
            x = src_img_feats.permute(0, 2, 3, 1)
            src_contrast_losses = []
            for batch_id in range(len(src_img_feats)):
                # pts_feats
                pts_feats = src_pts_feats[batch_id]  # (N, 64)

                # img_feats
                pts_indices = src_pts_indices[batch_id]
                img_feats = x[batch_id][pts_indices[:, 0], pts_indices[: 1]]  # (N, 64)

                num_pts = len(pts_feats)
                if num_pts > self.max_pts:
                    idx = np.random.choice(num_pts, self.max_pts, replace=False)
                    pts_feats = pts_feats[idx]
                    img_feats = img_feats[idx]

                src_loss = self.contrast_criterion(pts_feats, img_feats)
                src_contrast_losses.append(src_loss)

            # target consistency
            x = tgt_img_feats.permute(0, 2, 3, 1)
            tgt_contrast_losses = []
            for batch_id in range(len(tgt_img_feats)):
                pts_feats = tgt_pts_feats[batch_id]

                pts_indices = tgt_pts_indices[batch_id]
                img_feats = x[batch_id][pts_indices[:, 0], pts_indices[:, 1]]

                num_pts = len(pts_feats)
                if num_pts > self.max_pts:
                    idx = np.random.choice(num_pts, self.max_pts, replace=False)
                    pts_feats = pts_feats[idx]
                    img_feats = img_feats[idx]

                tgt_loss = self.contrast_criterion(pts_feats, img_feats)
                tgt_contrast_losses.append(tgt_loss)

            src_contrast_loss = self.lambda_contrast * torch.mean(torch.tensor(src_contrast_losses))
            tgt_contrast_loss = self.lambda_contrast * torch.mean(torch.tensor(tgt_contrast_losses))
            losses['src_contrast_loss'] = src_contrast_loss
            losses['tgt_contrast_loss'] = tgt_contrast_loss
            loss, log_vars = parse_losses(losses)
            num_samples = len(src_data_batch['img_metas'])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # after_train_iter callback
            self.log_buffer.update(log_vars, num_samples)
            self.call_hook('after_train_iter')  # optimizer hook && logger hook
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, data_batch, train_mode):
        assert self.batch_processor is None
        if train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer)
        if not isinstance(outputs, dict):
            raise TypeError('"model.train_step()" or "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        # assert isinstance(data_loaders, list)
        # assert mmcv.is_list_of(workflow, tuple)
        # assert len(data_loaders) == len(workflow)
        self._max_epochs = max_epochs
        self._max_iters = self._max_epochs * len(data_loaders[0])

        work_dir = self.work_dir
        self.logger.info('Start running, host: %s, work_dir: %s', get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            self.train(data_loaders[0], data_loaders[1])

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filename, dst_file)
