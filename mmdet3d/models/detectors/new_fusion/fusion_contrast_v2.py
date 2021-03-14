import copy
import mmcv
import torch
from mmcv.runner import force_fp32
import numpy as np
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.base import Base3DDetector


@DETECTORS.register_module()
class FusionContrastV2(Base3DDetector):
    # incorporate contrast loss, rather than put it in runner
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
                 pretrained=None,
                 pts_fc=[],
                 contrast_criterion=None,
                 max_pts=4096,
                 lambda_contrast=0.1):
        super(FusionContrastV2, self).__init__()

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
        if pts_neck:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if contrast_criterion:
            self.contrast_criterion = builder.build_loss(contrast_criterion)
            self.max_pts = max_pts
            self.lambda_contrast = lambda_contrast

        fc_layers = []
        for i, (in_c, out_c) in enumerate(zip(pts_fc[:-1], pts_fc[1:])):
            fc_layers.append(nn.Linear(in_c, out_c))
            if i == len(pts_fc) - 2:
                break
            fc_layers.append(nn.ReLU(inplace=True))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def extract_pts_feat(self, pts):
        pts_feats = []
        for p in pts:
            pts_feats.append(self.fc_layers(p[:, :3]))
        return pts_feats

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        input_shape = img.shape[-2:]
        for img_meta in img_metas:
            img_meta.update(input_shape=input_shape)

        img_feats = self.img_backbone(img)
        return img_feats

    def extract_det_feat(self, pts, pts_feats, img_feats, pts_indices, img_metas):
        '''

        Args:
            pts: list of tensor(N_t, 4); len == batch_size
            pts_feats: list of tensor(N_t, C); len == batch_size
            img_feats: tensor (B, C, H, W)
            pts_indices: list of tensor(M_t, 2)
            img_metas:

        Returns:

        '''
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][pts_indices[i][:, 0], pts_indices[i][:, 1]])
        concat_pts = []
        for i in range(len(pts)):
            concat_pts.append(torch.cat([pts[i][:, :3], pts_feats[i], sample_feats[i]], 1))

        voxels, num_points, coors = self.voxelize(concat_pts)  # voxels=(M, T=64, ndim=3+64+64); coors=(M, 4), 4:(batch_idx, z, y, x)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)  # (M, C=64); M=num of non-empty voxels
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)  # (N, C, H, W) = (4, 64, 200, 400)
        x = self.pts_backbone(x)  
        # tuple of tensor: 
        # ((4, 192, 100, 200), (4, 432, 50, 100), (4, 1008, 25, 50)) for regnetx_3.2gf
        # ((4, 168, 100, 200), (4, 408, 50, 100), (4, 912, 25, 50)) for regnetx_1.6gf
        if self.with_pts_neck:    # FPN
            x = self.pts_neck(x)  # tuple of tensor: ((4, 256, 100, 200), (4, 256, 50, 100), (4, 256, 25, 50))
        return x

    def extract_feat(self, points, pts_indices, img, img_metas):
        img_feats = self.extract_img_feat(img, img_metas)  # (N, 64, 225, 400)
        pts_feats = self.extract_pts_feat(points)
        det_feats = self.extract_det_feat(points, pts_feats, img_feats, pts_indices, img_feats)  # output of FPN
        return img_feats, det_feats

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
                      gt_bboxes_ignore=None,
                      target=False):
        '''
        Args:
            img: (4, 3, 225, 400)
            seg_points: list of tensor (N, 4); len == batch_size
            seg_pts_indices: list of tensor (N, 2); len == batch_size
            seg_label: list of tensor (N, ); len == batch_size
            points: list of tensor (M, 4); len == batch_size
            pts_indices: list of tensor (M, 2); len == batch_size
            gt_bboxes_3d: list of LiDARInstance3Dboxes; gt_bboxes_3d[i].tensor.shape=(num_gt_i, 9); len == batch_size
            gt_labels_3d: list of tensor (num_gt_i, ); len ==batch_size
            img_metas:
            gt_bboxes_ignore:

        Returns: losses
        '''
        losses = dict()

        # forward_fusion
        img_feats = self.extract_img_feat(img, img_metas)  # (N, 64, 225, 400)
        seg_pts_feats = self.extract_pts_feat(seg_points)
        pts_feats = self.extract_pts_feat(points)
        contrast_loss = self.forward_contrast(img_feats, pts_feats, pts_indices, target=target)
        losses.update(contrast_loss)

        if target:
            return losses

        # forward seg head
        seg_logits = self.img_seg_head(img_feats=img_feats, seg_pts=seg_pts_feats, seg_pts_indices=seg_pts_indices)
        losses_img = self.img_seg_head.loss(seg_logits, seg_label)
        losses.update(losses_img)

        # forward det
        det_feats = self.extract_det_feat(points, pts_feats, img_feats, pts_indices, img_feats)  # output of FPN
        losses_pts = self.forward_pts_train(det_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses

    def forward_contrast(self, img_feats, pts_feats_list, pts_indices_list, target=False):
        x = img_feats.permute(0, 2, 3, 1)
        contrast_losses = []
        for batch_id in range(len(img_feats)):
            # pts_feats
            pts_feats = pts_feats_list[batch_id]  # (N, 64)

            # img_feats
            pts_idx = pts_indices_list[batch_id]
            img_feats = x[batch_id][pts_idx[:, 0], pts_idx[:, 1]]  # (N, 64)

            num_pts = len(pts_feats)
            if num_pts > self.max_pts:
                idx = np.random.choice(num_pts, self.max_pts, replace=False)
                pts_feats = pts_feats[idx]
                img_feats = img_feats[idx]

            loss = self.contrast_criterion(pts_feats, img_feats)
            contrast_losses.append(loss)
        # contrast_loss = self.lambda_contrast * torch.mean(torch.tensor(contrast_losses))  # bug !!!!
        contrast_loss = self.lambda_contrast * sum(contrast_losses) / len(contrast_losses)
        loss_name = 'tgt_contrast_loss' if target else 'src_contrast_loss'
        return {loss_name: contrast_loss}


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        outs = self.pts_bbox_head(pts_feats)
        # outs = (x0, x1 ,x2);
        # x0: list of cls_score; cls_score = (N, num_anchors * num_classes, Hi, Wi); len(x0) = num_scale
        # x1: list of bbox_pred, bbox_pred = (N, num_anchors * bbox_code_size, Hi, Wi); len(x1) = num_scale
        # x2: list of dir_cls_pred; dir_cls_pred = (N, num_anchors * 2, Hi, Wi); len(x2) = num_scale
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # bbox_list: list of tuple; len(bbox_list) = test_batch_size
        # bbox_list[0]: (bboxes, scores, labels)
        # bboxes: LiDARInstance3DBoxes of tensor (N, 9); scores: tensor (N, ); labels: tensor (N, )
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # bbox_results: [dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)] on cpu
        # bboxes: LiDARInstance3DBoxes on cpu
        # scores: tensor (N, ); labels: tensor(N, )
        return bbox_results

    def simple_test(self, img, seg_points, seg_pts_indices, points, pts_indices, img_metas, rescale=False):
        """Test function without augmentaiton."""
        seg_pts_feats = self.extract_pts_feat(seg_points)
        img_feats, det_feats = self.extract_feat(points, pts_indices, img, img_metas)

        seg_logits = self.img_seg_head(img_feats=img_feats, seg_pts=seg_pts_feats, seg_pts_indices=seg_pts_indices)

        bbox_list = [dict() for i in range(len(img_metas))]  # len(bbox_list)=batch_size
        bbox_pts = self.simple_test_pts(det_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return seg_logits, bbox_list

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

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(FusionContrastV2, self).init_weights(pretrained)
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
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

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
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        return hasattr(self, 'pts_neck') and self.pts_neck is not None