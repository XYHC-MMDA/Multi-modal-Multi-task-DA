import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from mmcv.runner import BaseRunner, RUNNERS, save_checkpoint, get_host_info
# from .base_runner import BaseRunner
# from .builder import RUNNERS
# from .checkpoint import save_checkpoint
# from .utils import get_host_info
from mmdet3d.apis import parse_losses


@RUNNERS.register_module()
class DiscRunner(BaseRunner):
    def __init__(self,
                 model,
                 seg_discriminator,
                 det_discriminator,
                 seg_disc_optimizer,
                 det_disc_optimizer,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(DiscRunner, self).__init__(model,
                                         batch_processor,
                                         optimizer,
                                         work_dir,
                                         logger,
                                         meta,
                                         max_iters,
                                         max_epochs)
        self.seg_disc = seg_discriminator
        self.det_disc = det_discriminator
        self.seg_opt = seg_disc_optimizer
        self.det_opt = det_disc_optimizer

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

    def train(self, src_data_loader, tgt_data_loader, **kwargs):
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
            # train on src
            # ------------------------

            # train Discriminators
            # if acc(disc) > max: don't train disc
            losses, seg_fusion_feats, det_fusion_feats = self.model(**src_data_batch)  # losses: {'seg_loss'=}
            seg_disc_loss = self.seg_disc.loss(seg_fusion_feats, src=True)
            det_disc_loss = self.det_disc.loss(det_fusion_feats, src=True)
            disc_loss = seg_disc_loss + det_disc_loss

            self.seg_opt.zero_grad()
            self.det_opt.zero_grad()
            disc_loss.backward()
            self.seg_opt.step()
            self.det_opt.step()

            # train network
            # if acc(disc) > min: add GAN Loss
            losses['seg_domain_loss'] = self.seg_disc.loss(seg_fusion_feats, src=False)
            losses['det_domain_loss'] = self.det_disc.loss(det_fusion_feats, src=False)

            loss, log_vars = parse_losses(losses)
            num_samples = len(src_data_batch['img_metas'])
            self.log_buffer.update(log_vars, num_samples)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------------
            # train on tgt
            # ------------------------

            # train Discriminators
            seg_fusion_feats, det_fusion_feats = self.model.forward_fusion(**tgt_data_batch)
            seg_disc_loss = self.seg_disc.loss(seg_fusion_feats, src=False)
            det_disc_loss = self.det_disc.loss(det_fusion_feats, src=False)
            disc_loss = seg_disc_loss + det_disc_loss

            self.seg_opt.zero_grad()
            self.det_opt.zero_grad()
            disc_loss.backward()
            self.seg_opt.step()
            self.det_opt.step()

            # train network without task loss
            seg_domain_loss = self.seg_disc.loss(seg_fusion_feats, src=True)
            det_domain_loss = self.det_disc.loss(det_fusion_feats, src=True)
            domain_loss = seg_domain_loss + det_domain_loss

            self.optimizer.zero_grad()
            domain_loss.backward()
            self.optimizer.step()

            self.call_hook('after_train_iter')  # optimizer hook && logger hook
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

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

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

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

