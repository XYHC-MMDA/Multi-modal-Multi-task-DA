from torch import nn as nn
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors.base import Base3DDetector


@DETECTORS.register_module()
class SegFusion(Base3DDetector):
    def __init__(self,
                 img_backbone=None,
                 pts_backbone=None,
                 num_classes=None,
                 prelogits_dim=None,
                 class_weights=None,
                 pretrained=None,
                 normalize=True,
                 train_cfg=None,
                 test_cfg=None):
        super(SegFusion, self).__init__()

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        self.init_weights(pretrained=pretrained)

        self.seg_head = nn.Linear(prelogits_dim, num_classes)
        self.class_weights = torch.tensor(class_weights)
        self.normalize = normalize

    def get_scn_input(self, scn_coords):
        scn_input = []
        for idx in range(len(scn_coords)):
            coords = scn_coords[idx]
            num_pts = len(coords)
            batch_idxs = torch.zeros(num_pts, 1, dtype=torch.long).to(coords.device)
            locs = torch.cat([coords, batch_idxs], dim=1)
            feats = torch.ones(num_pts, 1).to(coords.device)
            scn_input.append([locs, feats])
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
        pts_feats = []
        for x in scn_input:
            pts_feats.append(self.pts_backbone(x))
        return pts_feats

    def extract_feat(self, img, scn_coords, seg_pts_indices):
        img_feats = self.extract_img_feat(img, seg_pts_indices)
        pts_feats = self.extract_pts_feat(scn_coords)
        return img_feats, pts_feats

    def forward_logits(self, sample_feats, pts_feats):
        sample_feats = torch.cat(sample_feats)
        pts_feats = torch.cat(pts_feats)
        if self.normalize:
            sample_feats = F.normalize(sample_feats)
            pts_feats = F.normalize(pts_feats)
        fusion_feats = torch.cat([sample_feats, pts_feats], dim=1)
        seg_logits = self.seg_head(fusion_feats)
        return seg_logits

    def forward_train(self,
                      img=None,
                      seg_points=None,
                      scn_coords=None,
                      seg_pts_indices=None,
                      seg_label=None,
                      img_metas=None):
        '''
        Args:
            img: (4, 3, 225, 400)
            seg_points: list of tensor (N, 4); len == batch_size
            seg_points: list of tensor (N, 3); len == batch_size; dtype=torch.long
            seg_pts_indices: list of tensor (N, 2); len == batch_size
            seg_label: list of tensor (N, ); len == batch_size

        Returns: losses
        '''
        # img, pts forward
        sample_feats, pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)

        # forward logits
        seg_logits = self.forward_logits(sample_feats, pts_feats)

        # seg head
        seg_label = torch.cat(seg_label)
        class_weights = self.class_weights.to(img.device)
        seg_loss = F.cross_entropy(seg_logits, seg_label, weight=class_weights)

        return dict(seg_loss=seg_loss)

    def simple_test(self, img, seg_points, seg_pts_indices, scn_coords):
        sample_feats, pts_feats = self.extract_feat(img, scn_coords, seg_pts_indices)
        seg_logits = self.forward_logits(sample_feats, pts_feats)
        return seg_logits

    def forward_test(self,
                     img=None,
                     seg_points=None,
                     seg_pts_indices=None,
                     scn_coords=None,
                     **kwargs):
        assert len(img) == 1
        return self.simple_test(img=img,
                                seg_points=seg_points,
                                seg_pts_indices=seg_pts_indices,
                                scn_coords=scn_coords)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(SegFusion, self).init_weights(pretrained)
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
    def with_img_backbone(self):
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_img_neck(self):
        return hasattr(self, 'img_neck') and self.img_neck is not None
