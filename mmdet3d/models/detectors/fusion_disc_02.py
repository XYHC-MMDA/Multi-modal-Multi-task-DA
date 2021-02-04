import torch
from mmcv.runner import force_fp32, auto_fp16
import torch.nn as nn
# from torch import nn as nn
from torch.nn import functional as F
import torch.distributed as dist

from mmdet3d.core import Box3DMode, bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder

from collections import OrderedDict
from mmdet.utils import get_root_logger
from mmcv.utils import print_log


# return img_feats before fusion
@DETECTORS.register_module()
class FusionDisc02(nn.Module):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 img_backbone=None,
                 img_seg_head=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FusionDisc02, self).__init__()

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_seg_head:
            self.img_seg_head = builder.build_head(img_seg_head)

        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.img_backbone(img)
        return img_feats

    def extract_pts_feat(self, pts, img_feats, pts_indices, img_metas):
        """Extract features of points."""
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][pts_indices[i][:, 0], pts_indices[i][:, 1]])
        concat_pts = []
        for i in range(len(pts)):
            concat_pts.append(torch.cat([pts[i], sample_feats[i]], 1))

        voxels, num_points, coors = self.voxelize(concat_pts)  # voxels=(M, T=64, ndim=4+64); coors=(M, 4), (batch_idx, z, y, x)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)  # (M, C=128); M=num of non-empty voxels
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)  # (N, C, H, W) = (4, 128, 200, 400)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, pts_indices, img, img_metas):
        img_feats = self.extract_img_feat(img, img_metas)  # (N, 64, 225, 400)
        x = self.extract_pts_feat(pts=points,
                                  img_feats=img_feats,
                                  pts_indices=pts_indices,
                                  img_metas=img_metas)
        return img_feats, x

    def forward_train(self,
                      img=None,
                      seg_points=None,
                      seg_pts_indices=None,
                      seg_label=None,
                      points=None,
                      pts_indices=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      img_metas=None,
                      gt_bboxes_ignore=None):
        # points: list of tensor; len(points)=batch_size; points[0].shape=(num_points, 4)
        # print('len:', len(gt_bboxes_3d))  # batch_size
        img_feats, pts_feats = self.extract_feat(points, pts_indices, img, img_metas)

        losses = dict()
        seg_logits = self.img_seg_head(img_feats=img_feats, seg_pts=seg_points, seg_pts_indices=seg_pts_indices)
        # seg_fusion_feats = self.img_seg_head.forward_fusion(img_feats, seg_points, seg_pts_indices)
        # seg_logits = self.img_seg_head.forward_logits(seg_fusion_feats)
        losses_img = self.img_seg_head.loss(seg_logits, seg_label)
        losses.update(losses_img)

        # pts_feats: tuple
        losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses, img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img, seg_points, seg_pts_indices, points, pts_indices, img_metas, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(points, pts_indices, img, img_metas)
        seg_logits = self.img_seg_head(img_feats=img_feats, seg_pts=seg_points, seg_pts_indices=seg_pts_indices)

        bbox_list = [dict() for i in range(len(img_metas))]  # len(bbox_list)=batch_size
        bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return seg_logits, bbox_list

    def forward_test(self,
                     img=None,
                     seg_points=None,
                     seg_pts_indices=None,
                     points=None,
                     pts_indices=None,
                     img_metas=None,
                     seg_label=None,
                     **kwargs):
        num_augs = len(points)
        assert num_augs == 1
        return self.simple_test(img=img[0],
                                seg_points=seg_points[0],
                                seg_pts_indices=seg_pts_indices[0],
                                points=points[0],
                                pts_indices=pts_indices[0],
                                img_metas=img_metas[0],
                                **kwargs)

    @auto_fp16(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
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

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self.parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained is not None:
            logger = get_root_logger()
            print_log(f'load model from: {pretrained}', logger=logger)
        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
            pts_pretrained = pretrained.get('pts', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)
        if self.with_pts_backbone:
            self.pts_backbone.init_weights(pretrained=pts_pretrained)

        if self.with_pts_bbox:
            self.pts_bbox_head.init_weights()

    @property
    def with_pts_bbox(self):
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_pts_neck(self):
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_voxel_encoder(self):
        return hasattr(self, 'voxel_encoder') and self.voxel_encoder is not None

    @property
    def with_middle_encoder(self):
        return hasattr(self, 'middle_encoder') and self.middle_encoder is not None
