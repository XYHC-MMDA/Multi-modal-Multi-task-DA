import os.path as osp
import platform
import shutil
import time
import warnings

import torch
import numpy as np

import mmcv
from mmcv.runner import BaseRunner, RUNNERS, save_checkpoint, get_host_info
# from .base_runner import BaseRunner
# from .builder import RUNNERS
# from .checkpoint import save_checkpoint
# from .utils import get_host_info
from mmdet3d.apis import parse_losses, set_requires_grad


def generate_patch_pair(img_size, patch_size):
    h = img_size[0] // patch_size[0]
    w = img_size[1] // patch_size[1]
    p1, p2 = np.random.choice(h * w, 2, replace=False)
    r1, c1 = p1 // w, p1 % w
    r2, c2 = p2 // w, p2 % w
    return r1, c1, r2, c2


def pts_in_patch(pts_idx, r, c, patch_size):
    '''

    Args:
        pts_idx: tensor (N, 2); (row, col)
        r1: int
        c1: int
        patch_size: [a, b]

    Returns:

    '''
    x1, x2 = r * patch_size[0], (r + 1) * patch_size[0]
    y1, y2 = c * patch_size[1], (c + 1) * patch_size[1]
    mask = (torch.tensor([x1, y1]) <= pts_idx) & (pts_idx < torch.tensor([x2, y2]))
    mask = mask[:, 0] & mask[:, 1]
    return mask


@RUNNERS.register_module()
class FusionConsisRunner(BaseRunner):
    def __init__(self,
                 model,
                 discriminator=None,
                 disc_optimizer=None,
                 lambda_GANLoss=1.0,
                 patch_size=[25, 25],
                 patch_pairs=5,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(FusionConsisRunner, self).__init__(model,
                                                 batch_processor,
                                                 optimizer,
                                                 work_dir,
                                                 logger,
                                                 meta,
                                                 max_iters,
                                                 max_epochs)
        self.disc = discriminator
        self.disc_opt = disc_optimizer
        self.lambda_GANLoss = lambda_GANLoss  # L = L_task + self.lambda_GANLoss * L_GAN
        self.patch_size = patch_size
        self.patch_pairs = patch_pairs

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
            losses, src_img_feats, src_pts_feats = self.model(**src_data_batch)
            tgt_img_feats, tgt_pts_feats = self.model(**tgt_data_batch, target=True)
            # img_feats: (N, C, H, W)=(4, 64, 225, 400); pts_feats: list of (N, C)

            # ------------------------
            # train Discriminators
            # ------------------------

            # set_requires_grad([self.disc], requires_grad=True)
            for _ in range(self.patch_pairs):
                pts_indices_list = src_data_batch['pts_indices'].data[0]
                r1, c1, r2, c2 = generate_patch_pair(src_img_feats.shape[-2:], self.patch_size)
                pts_feats_list1, pts_feats_list2 = [], []
                for i, pts_idx in enumerate(pts_indices_list):
                    mask1 = pts_in_patch(pts_idx, r1, c1, self.patch_size)
                    pts_feats_list1.append(src_pts_feats[i][mask1])
                    mask2 = pts_in_patch(pts_idx, r2, c2, self.patch_size)
                    pts_feats_list2.append(src_pts_feats[i][mask2])
                src_img_patch1 = src_img_feats[:, :,
                             r1 * self.patch_size[0]:(r1 + 1) * self.patch_size[0],
                             c1 * self.patch_size[1]:(c1 + 1) * self.patch_size[1]]
                src_img_patch2 = src_img_feats[:, :,
                             r2 * self.patch_size[0]:(r2 + 1) * self.patch_size[0],
                             c2 * self.patch_size[1]:(c2 + 1) * self.patch_size[1]]

            # ------------------------
            # train network on source: task loss + GANLoss
            # ------------------------
            set_requires_grad([self.disc], requires_grad=False)
            losses, seg_src_feats, det_src_feats = self.model(**src_data_batch)  # forward; losses: {'seg_loss'=}

            seg_disc_logits = self.seg_disc(seg_src_feats)  # (N, 2)
            seg_disc_pred = seg_disc_logits.max(1)[1]  # (N, ); cuda
            seg_label = torch.ones_like(seg_disc_pred, dtype=torch.long).cuda()
            seg_acc = (seg_disc_pred == seg_label).float().mean()
            acc_threshold = 0.6
            if seg_acc > acc_threshold:
                losses['seg_src_GANloss'] = self.lambda_GANLoss * self.seg_disc.loss(seg_disc_logits, src=False)

            det_disc_logits = self.det_disc(det_src_feats)  # (4, 2, 49, 99)
            det_disc_pred = det_disc_logits.max(1)[1]  # (4, 49, 99); cuda
            det_label = torch.ones_like(det_disc_pred, dtype=torch.long).cuda()
            det_acc = (det_disc_pred == det_label).float().mean()
            if det_acc > acc_threshold:
                losses['det_src_GANloss'] = self.lambda_GANLoss * self.det_disc.loss(det_disc_logits, src=False)

            loss, log_vars = parse_losses(losses)
            num_samples = len(src_data_batch['img_metas'])
            log_vars['seg_src_Dloss'] = log_seg_src_Dloss
            log_vars['det_src_Dloss'] = log_det_src_Dloss
            log_vars['seg_tgt_Dloss'] = log_seg_tgt_Dloss
            log_vars['det_tgt_Dloss'] = log_det_tgt_Dloss
            log_vars['seg_src_acc'] = seg_acc.item()
            log_vars['det_src_acc'] = det_acc.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ------------------------
            # train network on target: only GANLoss
            # ------------------------
            self.optimizer.zero_grad()
            seg_tgt_feats, det_tgt_feats = self.model.forward_fusion(**tgt_data_batch)

            seg_disc_logits = self.seg_disc(seg_tgt_feats)  # (N, 2)
            seg_disc_pred = seg_disc_logits.max(1)[1]  # (N, ); cuda
            seg_label = torch.zeros_like(seg_disc_pred, dtype=torch.long).cuda()
            seg_acc = (seg_disc_pred == seg_label).float().mean()
            tgt_GANloss = None
            if seg_acc > acc_threshold:
                seg_tgt_loss = self.lambda_GANLoss * self.seg_disc.loss(seg_disc_logits, src=True)
                tgt_GANloss = seg_tgt_loss
                log_vars['seg_tgt_GANloss'] = seg_tgt_loss.item()

            det_disc_logits = self.det_disc(det_tgt_feats)  # (4, 2, 49, 99)
            det_disc_pred = det_disc_logits.max(1)[1]  # (4, 49, 99); cuda
            det_label = torch.zeros_like(det_disc_pred, dtype=torch.long).cuda()
            det_acc = (det_disc_pred == det_label).float().mean()
            if det_acc > acc_threshold:
                det_tgt_loss = self.lambda_GANLoss * self.det_disc.loss(det_disc_logits, src=True)
                log_vars['det_tgt_GANloss'] = det_tgt_loss.item()
                if tgt_GANloss is None:
                    tgt_GANloss = det_tgt_loss
                else:
                    tgt_GANloss += det_tgt_loss

            log_vars['seg_tgt_acc'] = seg_acc.item()
            log_vars['det_tgt_acc'] = det_acc.item()
            if tgt_GANloss is not None:
                tgt_GANloss.backward()
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
