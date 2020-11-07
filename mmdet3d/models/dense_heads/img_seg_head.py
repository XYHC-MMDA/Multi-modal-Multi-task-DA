import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class ImageSegHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ImageSegHead, self).__init__()
        self.in_channels = in_channels  # feat_channels
        self.num_classes = num_classes
        self.linear = nn.Linear(in_channels, num_classes)

    def forward(self, img_feats, img_meta, img_indices):
        """

        Args:
            img_feats: img_feats
            img_meta: to calculate img_indices

        Returns: logits=(M, num_classes); M: total pts in a batch

        """
        x = img_feats.permute(0, 2, 3, 1)
        mask_feats = []
        for i in range(x.shape[0]):
            mask_feats.append(x[i][img_indices[i][:, 0], img_indices[i][:, 1]])
        # mask_feats[i].shape=(img_indices[i].shape[0], 64)
        mask_feats = torch.cat(mask_feats, 0)  # shape=(M, 64)
        seg_logits = self.linear(mask_feats)  # shape=(M, num_classes)
        return seg_logits

    def loss(self, img_feats, img_meta, img_indices, seg_label):
        seg_logits = self.forward(img_feats, img_meta, img_indices)
        # seg_label[0].device: cuda:0
        y = torch.cat(seg_label)  # shape=(M,); dtype=torch.uint8
        # y = y.type(torch.LongTensor).cuda()
        seg_loss = F.cross_entropy(seg_logits, y, weight=None)
        exit(0)
        return dict(seg_loss=seg_loss)

