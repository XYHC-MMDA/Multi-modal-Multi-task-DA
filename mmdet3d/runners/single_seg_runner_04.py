import os.path as osp
import platform
import shutil
import time
import torch
import mmcv
from mmcv.runner import BaseRunner, RUNNERS, save_checkpoint, get_host_info
from mmdet3d.apis import parse_losses, set_requires_grad


@RUNNERS.register_module()
class SingleSegRunner04(BaseRunner):
    def __init__(self, model,
                 seg_disc=None,
                 seg_opt=None,
                 lambda_GANLoss=0.0,
                 src_acc_threshold=1.0,
                 tgt_acc_threshold=0.6,
                 return_fusion_feats=True,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(SingleSegRunner04, self).__init__(model, batch_processor, optimizer, work_dir, logger, meta,
                                                max_iters, max_epochs)
        self.seg_disc = seg_disc
        self.seg_opt = seg_opt
        self.lambda_GANLoss = lambda_GANLoss  # L = L_task + self.lambda_GANLoss * L_GAN
        self.src_acc_threshold = src_acc_threshold
        self.tgt_acc_threshold = tgt_acc_threshold
        self.return_fusion_feats = return_fusion_feats

    @property
    def with_disc(self):
        return hasattr(self, 'seg_disc') and self.seg_disc is not None

    def get_feats(self, img_feats, fusion_feats):
        if self.return_fusion_feats:
            return fusion_feats
        else:
            return img_feats

    def train(self, src_data_loader, tgt_data_loader):
        self.model.train()
        self.mode = 'train'
        self.data_loader = src_data_loader
        if self.with_disc:
            self.seg_disc.train()
            tgt_data_iter = iter(tgt_data_loader)
            assert len(tgt_data_loader) <= len(src_data_loader)

        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        for i, src_data_batch in enumerate(self.data_loader):
            # before train iter
            self._inner_iter = i
            self.call_hook('before_train_iter')

            if self.with_disc:
                # fetch target data batch
                tgt_data_batch = next(tgt_data_iter, None)
                if tgt_data_batch is None:
                    tgt_data_iter = iter(tgt_data_loader)
                    tgt_data_batch = next(tgt_data_iter)

            # ------------------------
            # train Discriminators
            # ------------------------

            if self.with_disc:
                set_requires_grad(self.seg_disc, requires_grad=True)
                # seg_feats_before_fusion: (N, 64, 225, 400); seg_feats_after_fusion: (N, 128)
                # src segmentation
                src_img_feats, src_fusion_feats = self.model.extract_feat(**src_data_batch)
                src_feats = self.get_feats(src_img_feats, src_fusion_feats)
                src_logits = self.seg_disc(src_feats.detach())
                src_Dloss = self.seg_disc.loss(src_logits, src=True)
                log_src_Dloss = src_Dloss.item()

                # tgt segmentation
                tgt_img_feats, tgt_fusion_feats = self.model.extract_feat(**tgt_data_batch)
                tgt_feats = self.get_feats(tgt_img_feats, tgt_fusion_feats)
                tgt_logits = self.seg_disc(tgt_feats.detach())
                tgt_Dloss = self.seg_disc.loss(tgt_logits, src=False)
                log_tgt_Dloss = tgt_Dloss.item()

                Dloss = (src_Dloss + tgt_Dloss) * 0.5
                self.seg_opt.zero_grad()
                Dloss.backward()
                self.seg_opt.step()

            # ------------------------
            # network forward on source: task loss + self.lambda_GANLoss * GANLoss
            # ------------------------
            losses, src_img_feats, src_fusion_feats = self.model(**src_data_batch)  # forward; losses: {'seg_loss'=}
            src_feats = self.get_feats(src_img_feats, src_fusion_feats)

            if self.with_disc:
                set_requires_grad(self.seg_disc, requires_grad=False)
                disc_logits = self.seg_disc(src_feats)  # (N, 2)
                disc_pred = disc_logits.max(1)[1]  # (N, ); cuda
                seg_label = torch.ones_like(disc_pred, dtype=torch.long).cuda()
                src_acc = (disc_pred == seg_label).float().mean()
                if src_acc > self.src_acc_threshold:
                    losses['src_GANloss'] = self.lambda_GANLoss * self.seg_disc.loss(disc_logits, src=False)

            # ------------------------
            # network forward on target: only GANLoss
            # ------------------------
            if self.with_disc:
                tgt_img_feats, tgt_fusion_feats = self.model.extract_feat(**tgt_data_batch)
                tgt_feats = self.get_feats(tgt_img_feats, tgt_fusion_feats)

                disc_logits = self.seg_disc(tgt_feats)  # (N, 2)
                disc_pred = disc_logits.max(1)[1]  # (N, ); cuda
                seg_label = torch.zeros_like(disc_pred, dtype=torch.long).cuda()
                tgt_acc = (disc_pred == seg_label).float().mean()
                if tgt_acc > self.tgt_acc_threshold:
                    tgt_GANloss = self.lambda_GANLoss * self.seg_disc.loss(disc_logits, src=True)
                    losses['tgt_GANloss'] = tgt_GANloss  # original tgt_loss

            loss, log_vars = parse_losses(losses)
            num_samples = len(src_data_batch['img_metas'])
            if self.with_disc:
                log_vars['src_Dloss'] = log_src_Dloss
                log_vars['tgt_Dloss'] = log_tgt_Dloss
                log_vars['src_acc'] = src_acc.item()
                log_vars['tgt_acc'] = tgt_acc.item()

            # ------------------------
            # network backward: src_task_loss + self.lambda_GANLoss * tgt_GANloss
            # ------------------------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # after_train_iter callback
            self.log_buffer.update(log_vars, num_samples)
            self.call_hook('after_train_iter')  # optimizer hook && logger hook
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

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

