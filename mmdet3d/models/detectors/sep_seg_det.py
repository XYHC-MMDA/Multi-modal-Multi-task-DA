import copy
import mmcv
import torch
from mmcv.runner import force_fp32
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class SepSegDet(Base3DDetector):
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
        super(SepSegDet, self).__init__()

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
        voxels, num_points, coors = self.voxelize(pts)  # voxels=(M, T=64, ndim=4); coors=(M, 4), (batch_idx, z, y, x)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)  # (M, C=64); M=num of non-empty voxels
        batch_size = coors[-1, 0] + 1  # the first column of coord indicates which batch
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)  # (N, C, H, W) = (4, 64, 400, 400)
        x = self.pts_backbone(x)  # tuple of tensor: ((4, 64, 200, 200), (4, 128, 100, 100), (4, 256, 50, 50))
        if self.with_pts_neck:    # FPN
            x = self.pts_neck(x)  # tuple of tensor: ((4, 256, 200, 200), (4, 256, 100, 100), (4, 256, 50, 50))
        return x

    def extract_feat(self, points, pts_indices, img, img_metas):
        img_feats = self.extract_img_feat(img, img_metas)  # (N, 64, 225, 400)
        pts_feats = self.extract_pts_feat(pts=points, img_feats=img_feats, pts_indices=pts_indices, img_metas=img_metas)
        return img_feats, pts_feats

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
        seg_logits = self.img_seg_head(img_feats=img_feats, seg_pts_indices=seg_pts_indices)
        losses_img = self.img_seg_head.loss(seg_logits, seg_label)
        losses.update(losses_img)

        # pts_feats: tuple
        losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

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
        seg_logits = self.img_seg_head(img_feats=img_feats, seg_pts_indices=seg_pts_indices)

        bbox_list = [dict() for i in range(len(img_metas))]  # len(bbox_list)=batch_size
        bbox_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
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
        super(SepSegDet, self).init_weights(pretrained)
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
