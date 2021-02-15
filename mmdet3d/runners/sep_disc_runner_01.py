import os.path as osp
import platform
import shutil
import time
import torch
import mmcv
from mmcv.runner import BaseRunner, RUNNERS, save_checkpoint, get_host_info
from mmdet3d.apis import parse_losses, set_requires_grad


@RUNNERS.register_module()
class SepDiscRunner01(BaseRunner):
    def __init__(self, model,
                 img_disc=None,
                 img_opt=None,
                 lidar_disc=None,
                 lidar_opt=None,
                 lambda_img=0.0,
                 lambda_lidar=0.0,
                 src_acc_threshold=1.0,
                 tgt_acc_threshold=0.6,
                 return_fusion_feats=False,  # True: fusion feats; False: lidar feats
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(SepDiscRunner01, self).__init__(model, batch_processor, optimizer, work_dir, logger, meta,
                                              max_iters, max_epochs)
        self.img_disc = img_disc
        self.img_opt = img_opt
        self.lidar_disc = lidar_disc
        self.lidar_opt = lidar_opt
        self.lambda_img = lambda_img
        self.lambda_lidar = lambda_lidar
        self.src_acc_threshold = src_acc_threshold
        self.tgt_acc_threshold = tgt_acc_threshold
        self.return_fusion_feats = return_fusion_feats

    def get_feats(self, lidar_feats, fusion_feats):
        if self.return_fusion_feats:
            return fusion_feats
        else:
            return lidar_feats

    def train(self, src_data_loader, tgt_data_loader):
        self.model.train()
        self.img_disc.train()
        self.lidar_disc.train()
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
            # train Discriminators
            # ------------------------

            set_requires_grad([self.img_disc, self.lidar_disc], requires_grad=True)
            # img_feats: (N, 64, 225, 400); lidar_feats: (N, 64); fusion_feats: (N, 128)
            # src
            src_img_feats, src_lidar_feats, src_fusion_feats = self.model.extract_feat(**src_data_batch)
            src_feats0 = src_img_feats
            src_feats1 = self.get_feats(src_lidar_feats, src_fusion_feats)

            src_logits0 = self.img_disc(src_feats0.detach())
            src_Dloss0 = self.img_disc.loss(src_logits0, src=True)
            log_src_Dloss0 = src_Dloss0.item()

            src_logits1 = self.lidar_disc(src_feats1.detach())
            src_Dloss1 = self.lidar_disc.loss(src_logits1, src=True)
            log_src_Dloss1 = src_Dloss1.item()

            # tgt segmentation
            tgt_img_feats, tgt_lidar_feats, tgt_fusion_feats = self.model.extract_feat(**tgt_data_batch)
            tgt_feats0 = tgt_img_feats
            tgt_feats1 = self.get_feats(tgt_lidar_feats, tgt_fusion_feats)

            tgt_logits0 = self.img_disc(tgt_feats0.detach())
            tgt_Dloss0 = self.img_disc.loss(tgt_logits0, src=False)
            log_tgt_Dloss0 = tgt_Dloss0.item()

            tgt_logits1 = self.lidar_disc(tgt_feats1.detach())
            tgt_Dloss1 = self.lidar_disc.loss(tgt_logits1, src=False)
            log_tgt_Dloss1 = tgt_Dloss1.item()

            # backward
            img_Dloss = (src_Dloss0 + tgt_Dloss0) * 0.5
            lidar_Dloss = (src_Dloss1 + tgt_Dloss1) * 0.5

            self.img_opt.zero_grad()
            img_Dloss.backward()
            self.img_opt.step()

            self.lidar_opt.zero_grad()
            lidar_Dloss.backward()
            self.lidar_opt.step()

            # ------------------------
            # network forward on source: task loss + self.lambda_GANLoss * GANLoss
            # ------------------------
            losses, src_img_feats, src_lidar_feats, src_fusion_feats = self.model(**src_data_batch)
            src_feats0 = src_img_feats
            src_feats1 = self.get_feats(src_lidar_feats, src_fusion_feats)

            set_requires_grad([self.img_disc, self.lidar_disc], requires_grad=False)
            src_logits0 = self.img_disc(src_feats0)
            src_pred0 = src_logits0.max(1)[1]
            src_label0 = torch.ones_like(src_pred0, dtype=torch.long).cuda()
            src_acc0 = (src_pred0 == src_label0).float().mean()
            if src_acc0 > self.src_acc_threshold:
                losses['src_img_GANloss'] = self.lambda_img * self.img_disc.loss(src_logits0, src=False)

            src_logits1 = self.lidar_disc(src_feats1)
            src_pred1 = src_logits1.max(1)[1]
            src_label1 = torch.ones_like(src_pred1, dtype=torch.long).cuda()
            src_acc1 = (src_pred1 == src_label1).float().mean()
            if src_acc1 > self.src_acc_threshold:
                losses['src_lidar_GANloss'] = self.lambda_lidar * self.lidar_disc.loss(src_logits1, src=False)
            # ------------------------
            # network forward on target: only GANLoss
            # ------------------------
            tgt_img_feats, tgt_lidar_feats, tgt_fusion_feats = self.model.extract_feat(**tgt_data_batch)
            tgt_feats0 = tgt_img_feats
            tgt_feats1 = self.get_feats(tgt_img_feats, tgt_fusion_feats)

            tgt_logits0 = self.img_disc(tgt_feats0)
            tgt_pred0 = tgt_logits0.max(1)[1]
            tgt_label0 = torch.zeros_like(tgt_pred0, dtype=torch.long).cuda()
            tgt_acc0 = (tgt_pred0 == tgt_label0).float().mean()
            if tgt_acc0 > self.tgt_acc_threshold:
                losses['tgt_img_GANloss'] = self.lambda_img * self.img_disc.loss(tgt_logits0, src=True)

            tgt_logits1 = self.lidar_disc(tgt_feats1)
            tgt_pred1 = tgt_logits1.max(1)[1]
            tgt_label1 = torch.zeros_like(tgt_pred1, dtype=torch.long).cuda()
            tgt_acc1 = (tgt_pred1 == tgt_label1).float().mean()
            if tgt_acc1 > self.tgt_acc_threshold:
                losses['tgt_lidar_GANloss'] = self.lambda_lidar * self.lidar_disc.loss(tgt_logits1, src=True)

            loss, log_vars = parse_losses(losses)
            num_samples = len(src_data_batch['img_metas'])
            log_vars['src_img_Dloss'] = log_src_Dloss0
            log_vars['src_lidar_Dloss'] = log_src_Dloss1
            log_vars['tgt_img_Dloss'] = log_tgt_Dloss0
            log_vars['tgt_lidar_Dloss'] = log_tgt_Dloss1
            log_vars['src_img_acc'] = src_acc0.item()
            log_vars['src_lidar_acc'] = src_acc1.item()
            log_vars['tgt_img_acc'] = tgt_acc0.item()
            log_vars['tgt_lidar_acc'] = tgt_acc1.item()

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

