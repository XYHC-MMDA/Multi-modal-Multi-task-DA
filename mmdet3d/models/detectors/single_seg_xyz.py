from torch import nn as nn
from mmdet.models import DETECTORS
from .. import builder
from .base import Base3DDetector


@DETECTORS.register_module()
class SingleSegXYZ(Base3DDetector):
    def __init__(self,
                 img_backbone=None,
                 img_seg_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleSegXYZ, self).__init__()

        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_seg_head:
            self.img_seg_head = builder.build_head(img_seg_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def extract_img_feat(self, img, img_metas=None):
        """Extract features of images."""
        # input_shape = img.shape[-2:]
        # for img_meta in img_metas:
        #     img_meta.update(input_shape=input_shape)
        img_feats = self.img_backbone(img)
        return img_feats

    def extract_fusion_feats(self, img, seg_points, seg_pts_indices):
        img_feats = self.extract_img_feat(img)
        fusion_feats = self.img_seg_head.forward_fusion(img_feats, seg_points, seg_pts_indices)
        return img_feats, fusion_feats

    def forward_train(self,
                      img=None,
                      seg_points=None,
                      seg_pts_indices=None,
                      seg_label=None,
                      img_metas=None):
        # img_feats = self.extract_img_feat(img, img_metas)
        # fusion_feats = self.img_seg_head.forward_fusion(img_feats, seg_points, seg_pts_indices)
        img_feats, fusion_feats = self.extract_fusion_feats(img, seg_points, seg_pts_indices)
        seg_logits = self.img_seg_head.forward_logits(fusion_feats)
        losses_img = self.img_seg_head.loss(seg_logits, seg_label)  # key='seg_loss'
        return losses_img, img_feats, fusion_feats

    def simple_test(self, img, seg_points, seg_pts_indices):
        """Test function without augmentaiton."""
        img_feats, fusion_feats = self.extract_fusion_feats(img, seg_points, seg_pts_indices)
        seg_logits = self.img_seg_head.forward_logits(fusion_feats)
        return seg_logits

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(SingleSegXYZ, self).init_weights(pretrained)
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
