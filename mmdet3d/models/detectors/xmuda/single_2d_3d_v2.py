from torch import nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.apis import build_mlp


@DETECTORS.register_module()
class Single2D3DV2(Base3DDetector):
    # batch input for 3d network instead of for loop
    def __init__(self,
                 img_backbone=None,
                 pts_backbone=None,
                 num_classes=None,
                 prelogits_dim=None,
                 class_weights=None,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None):
        super(Single2D3DV2, self).__init__()

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        self.init_weights(pretrained=pretrained)

        self.seg_head = nn.Linear(prelogits_dim, num_classes)
        self.class_weights = torch.tensor(class_weights)

    def get_scn_input(self, scn_coords):
        locs = []
        feats = []
        for idx in range(len(scn_coords)):
            coords = scn_coords[idx]
            num_pts = len(coords)
            batch_idxs = torch.LongTensor(num_pts, 1).fill_(idx).to(coords.device)
            # batch_idxs = torch.zeros(num_pts, 1, dtype=torch.long).to(coords.device)
            locs.append(torch.cat([coords, batch_idxs], dim=1))
            feats.append(torch.ones(num_pts, 1).to(coords.device))
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        scn_input = [locs, feats]
        return scn_input

    def extract_img_feat(self, img, seg_pts_indices):
        img_feats = self.img_backbone(img)
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])
        return sample_feats

    def extract_pts_feat(self, scn_coords):
        scn_input = self.get_scn_input(scn_coords)
        pts_feats = self.pts_backbone(scn_input)
        return pts_feats

    def extract_feat(self, img, scn_coords, seg_pts_indices):
        sample_feats, pts_feats = None, None
        if self.with_img_backbone:
            sample_feats = self.extract_img_feat(img, seg_pts_indices)
        if self.with_pts_backbone:
            pts_feats = self.extract_pts_feat(scn_coords)
        return sample_feats, pts_feats

    def forward_logits(self, sample_feats, pts_feats):
        if self.with_img_backbone and self.with_pts_backbone:
            sample_feats = torch.cat(sample_feats)
            fusion_feats = torch.cat([sample_feats, pts_feats], dim=1)
            seg_logits = self.seg_head(fusion_feats)
        elif self.with_img_backbone:
            sample_feats = torch.cat(sample_feats)
            seg_logits = self.seg_head(sample_feats)
        elif self.with_pts_backbone:
            seg_logits = self.seg_head(pts_feats)
        else:
            assert False
        return seg_logits

    def forward_train(self,
                      img=None,
                      seg_points=None,
                      scn_coords=None,
                      seg_pts_indices=None,
                      seg_label=None,
                      img_metas=None
                      ):
        '''
        Args:
            img: (4, 3, 225, 400)
            seg_points: list of tensor (N, 4); len == batch_size
            seg_points: list of tensor (N, 3); len == batch_size; dtype=torch.long
            seg_pts_indices: list of tensor (N, 2); len == batch_size
            seg_label: list of tensor (N, ); len == batch_size

        Returns: losses
        '''
        losses = dict()

        # img, pts forward
        sample_feats, pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)
        # sample_feats: list of (N_i, 64) or None; pts_feats: (sigma(N_i), 16) or None

        # forward logits
        seg_logits = self.forward_logits(sample_feats, pts_feats)

        # seg loss
        seg_label = torch.cat(seg_label)
        class_weights = self.class_weights.to(img.device)
        seg_loss = F.cross_entropy(seg_logits, seg_label, weight=class_weights)
        seg_loss_dict = dict(seg_loss=seg_loss)
        losses.update(seg_loss_dict)

        return losses

    def simple_test(self, img, seg_label, seg_pts_indices, scn_coords, with_loss):
        # sample_feats, pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)  # huge bug: wrong order
        sample_feats, pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)
        seg_logits = self.forward_logits(sample_feats, pts_feats)
        if not with_loss:
            return seg_logits
        else:
            seg_label = torch.cat(seg_label)
            class_weights = self.class_weights.to(img.device)
            seg_loss = F.cross_entropy(seg_logits, seg_label, weight=class_weights)
            return seg_logits, seg_loss

    def forward_test(self,
                     img=None,
                     seg_points=None,
                     seg_label=None,
                     seg_pts_indices=None,
                     scn_coords=None,
                     with_loss=False,
                     **kwargs):
        # assert len(img) == 1
        return self.simple_test(img=img,
                                seg_label=seg_label,
                                seg_pts_indices=seg_pts_indices,
                                scn_coords=scn_coords,
                                with_loss=with_loss)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(Single2D3DV2, self).init_weights(pretrained)
        if pretrained is None:
            img_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None
