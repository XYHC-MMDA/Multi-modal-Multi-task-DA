from torch import nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.apis import build_mlp


@DETECTORS.register_module()
class SegFusionContra(Base3DDetector):
    # batch input for 3d network instead of for loop
    def __init__(self,
                 img_backbone=None,
                 pts_backbone=None,
                 num_classes=None,
                 prelogits_dim=None,
                 class_weights=None,
                 pretrained=None,
                 contrast_criterion=None,
                 max_pts=1024,  # max_pts_per_group
                 groups=1,  # number of groups per sample
                 lambda_contrast=0.1,
                 img_fcs=(64, 64, 16),
                 pts_fcs=(16, 16, 16),
                 train_cfg=None,
                 test_cfg=None):
        super(SegFusionContra, self).__init__()

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        self.init_weights(pretrained=pretrained)

        self.seg_head = nn.Linear(prelogits_dim, num_classes)
        self.class_weights = torch.tensor(class_weights)
        if contrast_criterion:
            self.contrast_criterion = builder.build_loss(contrast_criterion)
            self.max_pts = max_pts
            self.groups = groups
            self.lambda_contrast = lambda_contrast
            self.img_fc = build_mlp(img_fcs)
            self.pts_fc = build_mlp(pts_fcs)

    def get_scn_input(self, scn_coords):
        locs = []
        feats = []
        for idx in range(len(scn_coords)):
            coords = scn_coords[idx]
            num_pts = len(coords)
            batch_idxs = torch.LongTensor(num_pts, 1).fill_(idx).to(coords.device)
            locs.append(torch.cat([coords, batch_idxs], dim=1))
            feats.append(torch.ones(num_pts, 1).to(coords.device))
        locs = torch.cat(locs, 0)
        feats = torch.cat(feats, 0)
        scn_input = [locs, feats]
        return scn_input

    def extract_img_feat(self, img, seg_pts_indices):
        (x_m, scale), img_feats = self.img_backbone(img)

        # last feature map
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])

        # intermediate feature map
        x_m = x_m.permute(0, 2, 3, 1)
        inter_img_feats = []
        for i in range(x_m.shape[0]):
            inter_img_feats.append(x_m[i][seg_pts_indices[i][:, 0] // scale, seg_pts_indices[i][:, 1] // scale])
        return inter_img_feats, sample_feats

    def extract_pts_feat(self, scn_coords):
        scn_input = self.get_scn_input(scn_coords)
        inter_pts_feats, pts_feats = self.pts_backbone(scn_input)
        return inter_pts_feats, pts_feats

    def extract_feat(self, img, scn_coords, seg_pts_indices):
        inter_img_feats, img_feats = self.extract_img_feat(img, seg_pts_indices)
        inter_pts_feats, pts_feats = self.extract_pts_feat(scn_coords)
        return inter_img_feats, img_feats, inter_pts_feats, pts_feats

    def forward_logits(self, sample_feats, pts_feats):
        sample_feats = torch.cat(sample_feats)
        pts_feats = torch.cat(pts_feats)
        fusion_feats = torch.cat([sample_feats, pts_feats], dim=1)
        seg_logits = self.seg_head(fusion_feats)
        return seg_logits

    def forward_train(self,
                      img=None,
                      seg_points=None,
                      scn_coords=None,
                      seg_pts_indices=None,
                      seg_label=None,
                      img_metas=None,
                      only_contrast=False,
                      freeze=False):
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
        inter_sample_feats, sample_feats, concat_inter_pts_feats, concat_pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)
        # sample_feats: list of (N_i, 64); concat_pts_feats: (sigma(N_i), 16)
        inter_pts_feats, pts_feats = [], []
        l = 0
        for label_i in seg_label:
            num_pts = len(label_i)
            pts_feats.append(concat_pts_feats[l:l+num_pts])
            inter_pts_feats.append(concat_inter_pts_feats[l:l+num_pts])
            l += num_pts
        # assert l == len(concat_pts_feats)

        # forward fusion; assert self.with_contrast or not only_contrast
        if self.with_contrast:
            contrast_loss = self.forward_contrast(inter_sample_feats, inter_pts_feats)
            losses.update(contrast_loss)
        elif only_contrast:
            assert False
        if only_contrast:
            return losses

        # forward logits
        seg_logits = self.forward_logits(sample_feats, pts_feats)

        # seg loss
        seg_label = torch.cat(seg_label)
        class_weights = self.class_weights.to(img.device)
        seg_loss = F.cross_entropy(seg_logits, seg_label, weight=class_weights)
        seg_loss_dict = dict(seg_loss=seg_loss)
        losses.update(seg_loss_dict)

        return losses

    def forward_contrast(self, img_feats_list, pts_feats_list):
        contrast_losses = []
        for batch_id in range(len(img_feats_list)):
            img_feats = img_feats_list[batch_id]  # (N, 64)
            pts_feats = pts_feats_list[batch_id]  # (N, 16)

            num_pts = len(pts_feats)
            for group_idx in range(self.groups):
                if num_pts > self.max_pts:
                    idx = np.random.choice(num_pts, self.max_pts, replace=False)
                    gi_img_feats = img_feats[idx]
                    gi_pts_feats = pts_feats[idx]
                else:
                    gi_img_feats = img_feats
                    gi_pts_feats = pts_feats

                feats_1 = self.img_fc(gi_img_feats)
                feats_2 = self.pts_fc(gi_pts_feats)
                loss = self.contrast_criterion(feats_1, feats_2)
                contrast_losses.append(loss)
        # len(contrast_losses) == self.groups * batch_size
        contrast_loss = self.lambda_contrast * sum(contrast_losses) / len(contrast_losses)
        return dict(contrast_loss=contrast_loss)

    def simple_test(self, img, seg_label, seg_pts_indices, scn_coords, with_loss):
        sample_feats, concat_pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)
        pts_feats = []
        l = 0
        for label_i in seg_label:
            num_pts = len(label_i)
            pts_feats.append(concat_pts_feats[l:l+num_pts])
            l += num_pts
        assert l == len(concat_pts_feats)

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
        assert len(img) == 1
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
        super(SegFusionContra, self).init_weights(pretrained)
        if pretrained is None:
            img_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get('img', None)
        else:
            raise ValueError(
                f'pretrained should be a dict, got {type(pretrained)}')
        if self.with_img_backbone:
            self.img_backbone.init_weights(pretrained=img_pretrained)
        if self.with_img_neck:
            if isinstance(self.img_neck, nn.Sequential):
                for m in self.img_neck:
                    m.init_weights()
            else:
                self.img_neck.init_weights()

    @property
    def with_contrast(self):
        return hasattr(self, 'contrast_criterion') and self.contrast_criterion is not None

    @property
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None
