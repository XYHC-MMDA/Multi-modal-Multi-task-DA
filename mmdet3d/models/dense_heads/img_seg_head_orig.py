import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class ImageSegHead(nn.Module):
    def __init__(self, img_feat_dim, seg_pts_dim, num_classes, lidar_fc=[], concat_fc=[]):
        super(ImageSegHead, self).__init__()
        # self.in_channels = in_channels  # feat_channels
        self.num_classes = num_classes
        self.lidar_fc = [seg_pts_dim] + lidar_fc
        self.concat_fc = [self.lidar_fc[-1] + img_feat_dim] + concat_fc

        self.before_fusion = []
        for i, (in_dim, out_dim) in enumerate(zip(self.lidar_fc[:-1], self.lidar_fc[1:])):
            self.before_fusion.append(nn.Linear(in_dim, out_dim))
            if i == len(lidar_fc) - 1:  # do not add activation in the last layer
                break
            self.before_fusion.append(nn.ReLU(inplace=True))

        self.after_fusion = []
        for i, (in_dim, out_dim) in enumerate(zip(self.concat_fc[:-1], self.concat_fc[1:])):
            self.after_fusion.append(nn.Linear(in_dim, out_dim))
            self.after_fusion.append(nn.ReLU(inplace=True))

        self.head = nn.Linear(self.concat_fc[-1], num_classes)

    def forward(self, img_feats, seg_pts, seg_pts_indices, img_metas=None):
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])
        # sample_feats[i].shape=(img_indices[i].shape[0], 64)
        sample_feats = torch.cat(sample_feats)  # shape=(M, 64)

        lidar_feat = torch.cat(seg_pts)  # (M, pts_dim=4)
        for layer in self.before_fusion:
            lidar_feat = layer(lidar_feat)

        concat_feats = torch.cat([sample_feats, lidar_feat], 1)  # (M, 64 + C)
        for layer in self.after_fusion:
            concat_feats = layer(concat_feats)

        seg_logits = self.head(concat_feats)  # (M, num_classes)
        return seg_logits

    def loss(self, seg_logits, seg_label):
        # seg_logits = self.forward(img_feats, img_indices, img_meta)
        # seg_label[0].device: cuda:0
        y = torch.cat(seg_label)  # shape=(M,); dtype=torch.uint8
        # y = y.type(torch.LongTensor).cuda()
        seg_loss = F.cross_entropy(seg_logits, y, weight=None)
        return dict(seg_loss=seg_loss)

