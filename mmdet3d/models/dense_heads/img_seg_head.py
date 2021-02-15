import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS


@HEADS.register_module()
class ImageSegHead(nn.Module):
    def __init__(self, img_feat_dim, seg_pts_dim, num_classes, lidar_fc=[], concat_fc=[], class_weights=None):
        super(ImageSegHead, self).__init__()
        # self.in_channels = in_channels  # feat_channels
        self.num_classes = num_classes
        self.lidar_fc = [seg_pts_dim] + lidar_fc
        self.concat_fc = [self.lidar_fc[-1] + img_feat_dim] + concat_fc
        if class_weights:
            self.class_weights = torch.tensor(class_weights).cuda()
        else:
            self.class_weights = None

        before_fusion = []
        for i, (in_dim, out_dim) in enumerate(zip(self.lidar_fc[:-1], self.lidar_fc[1:])):
            before_fusion.append(nn.Linear(in_dim, out_dim))
            if i == len(lidar_fc) - 1:  # do not add activation in the last layer
                break
            before_fusion.append(nn.ReLU(inplace=True))
        self.before_fusion = nn.Sequential(*before_fusion)

        after_fusion = []
        for i, (in_dim, out_dim) in enumerate(zip(self.concat_fc[:-1], self.concat_fc[1:])):
            after_fusion.append(nn.Linear(in_dim, out_dim))
            after_fusion.append(nn.ReLU(inplace=True))
        self.after_fusion = nn.Sequential(*after_fusion)

        self.head = nn.Linear(self.concat_fc[-1], num_classes)

    def forward_fusion(self, img_feats, seg_pts, seg_pts_indices):
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])
        # sample_feats[i].shape=(img_indices[i].shape[0], 64)
        sample_feats = torch.cat(sample_feats)  # shape=(M, 64); M=total points in a batch

        lidar_feat = torch.cat(seg_pts)  # (M, pts_dim=4)
        lidar_feat = self.before_fusion(lidar_feat)

        fusion_feats = torch.cat([sample_feats, lidar_feat], 1)  # (M, 64 + C)
        return fusion_feats

    def forward_logits(self, fusion_feats):
        fusion_feats = self.after_fusion(fusion_feats)
        seg_logits = self.head(fusion_feats)  # (M, num_classes)
        return seg_logits

    def forward(self, img_feats, seg_pts, seg_pts_indices):
        fusion_feats = self.forward_fusion(img_feats, seg_pts, seg_pts_indices)
        seg_logits = self.forward_logits(fusion_feats)
        return seg_logits

    def loss(self, seg_logits, seg_label):
        # seg_logits = self.forward(img_feats, img_indices, img_meta)
        # seg_label[0].device: cuda:0
        y = torch.cat(seg_label)  # shape=(M,); dtype=torch.uint8
        # y = y.type(torch.LongTensor).cuda()
        seg_loss = F.cross_entropy(seg_logits, y, weight=self.class_weights)
        return dict(seg_loss=seg_loss)


@HEADS.register_module()
class SepDiscHead(nn.Module):
    def __init__(self, img_feat_dim, seg_pts_dim, num_classes, lidar_fc=[], concat_fc=[], class_weights=None):
        super(SepDiscHead, self).__init__()
        # self.in_channels = in_channels  # feat_channels
        self.num_classes = num_classes
        self.lidar_fc = [seg_pts_dim] + lidar_fc
        self.concat_fc = [self.lidar_fc[-1] + img_feat_dim] + concat_fc
        if class_weights:
            self.class_weights = torch.tensor(class_weights).cuda()
        else:
            self.class_weights = None

        before_fusion = []
        for i, (in_dim, out_dim) in enumerate(zip(self.lidar_fc[:-1], self.lidar_fc[1:])):
            before_fusion.append(nn.Linear(in_dim, out_dim))
            if i == len(lidar_fc) - 1:  # do not add activation in the last layer
                break
            before_fusion.append(nn.ReLU(inplace=True))
        self.before_fusion = nn.Sequential(*before_fusion)

        after_fusion = []
        for i, (in_dim, out_dim) in enumerate(zip(self.concat_fc[:-1], self.concat_fc[1:])):
            after_fusion.append(nn.Linear(in_dim, out_dim))
            after_fusion.append(nn.ReLU(inplace=True))
        self.after_fusion = nn.Sequential(*after_fusion)

        self.head = nn.Linear(self.concat_fc[-1], num_classes)

    def forward_fusion(self, img_feats, seg_pts, seg_pts_indices):
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])
        # sample_feats[i].shape=(img_indices[i].shape[0], 64)
        sample_feats = torch.cat(sample_feats)  # shape=(M, 64); M=total points in a batch

        lidar_feats = torch.cat(seg_pts)  # (M, pts_dim=4)
        lidar_feats = self.before_fusion(lidar_feats)

        fusion_feats = torch.cat([sample_feats, lidar_feats], 1)  # (M, 64 + C)
        return lidar_feats, fusion_feats

    def forward_logits(self, fusion_feats):
        fusion_feats = self.after_fusion(fusion_feats)
        seg_logits = self.head(fusion_feats)  # (M, num_classes)
        return seg_logits

    def forward(self, img_feats, seg_pts, seg_pts_indices):
        fusion_feats = self.forward_fusion(img_feats, seg_pts, seg_pts_indices)
        seg_logits = self.forward_logits(fusion_feats)
        return seg_logits

    def loss(self, seg_logits, seg_label):
        # seg_logits = self.forward(img_feats, img_indices, img_meta)
        # seg_label[0].device: cuda:0
        y = torch.cat(seg_label)  # shape=(M,); dtype=torch.uint8
        # y = y.type(torch.LongTensor).cuda()
        seg_loss = F.cross_entropy(seg_logits, y, weight=self.class_weights)
        return dict(seg_loss=seg_loss)


@HEADS.register_module()
class ImageSegHeadWoFusion(nn.Module):
    def __init__(self, img_feat_dim, num_classes, class_weights=None):
        super(ImageSegHeadWoFusion, self).__init__()
        self.num_classes = num_classes
        if class_weights:
            self.class_weights = torch.tensor(class_weights).cuda()
        else:
            self.class_weights = None
        self.head = nn.Linear(img_feat_dim, num_classes)

    def forward(self, img_feats, seg_pts_indices):
        x = img_feats.permute(0, 2, 3, 1)
        sample_feats = []
        for i in range(x.shape[0]):
            sample_feats.append(x[i][seg_pts_indices[i][:, 0], seg_pts_indices[i][:, 1]])
        # sample_feats[i].shape=(img_indices[i].shape[0], 64)
        sample_feats = torch.cat(sample_feats)  # shape=(M, 64)

        seg_logits = self.head(sample_feats)  # (M, num_classes)
        return seg_logits

    def loss(self, seg_logits, seg_label):
        y = torch.cat(seg_label)  # shape=(M,); dtype=torch.uint8
        seg_loss = F.cross_entropy(seg_logits, y, weight=self.class_weights)
        return dict(seg_loss=seg_loss)
