import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class ImageSegHead(nn.Module):
    def __init__(self, img_feat_dim, seg_pts_dim, num_classes):
        super(ImageSegHead, self).__init__()
        # self.in_channels = in_channels  # feat_channels
        self.num_classes = num_classes
        self.linear = nn.Linear(img_feat_dim + seg_pts_dim, num_classes)

    def forward(self, img_feats, seg_pts, seg_pts_indices, img_metas=None):
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])
        # sample_feats[i].shape=(img_indices[i].shape[0], 64)
        sample_feats = torch.cat(sample_feats)  # shape=(M, 64)
        seg_pts_batch = torch.cat(seg_pts)  # (M, pts_dim=4)
        concat_feats = torch.cat([sample_feats, seg_pts_batch], 1)
        seg_logits = self.linear(concat_feats)  # shape=(M, num_classes)
        return seg_logits

    def loss(self, seg_logits, seg_label):
        # seg_logits = self.forward(img_feats, img_indices, img_meta)
        # seg_label[0].device: cuda:0
        y = torch.cat(seg_label)  # shape=(M,); dtype=torch.uint8
        # y = y.type(torch.LongTensor).cuda()
        seg_loss = F.cross_entropy(seg_logits, y, weight=None)
        return dict(seg_loss=seg_loss)

