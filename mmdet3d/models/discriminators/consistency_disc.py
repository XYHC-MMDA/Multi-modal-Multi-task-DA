import torch.nn as nn
import torch
import torch.nn.functional as F
from ..registry import DISCRIMINATORS


@DISCRIMINATORS.register_module()
class ConsistencyDisc(nn.Module):
    def __init__(self, fc=[128, 64], min_pts_threshold=10):
        super(ConsistencyDisc, self).__init__()
        self.min_pts_threshold = min_pts_threshold
        assert isinstance(fc, (tuple, list))
        fc_layers = []
        for in_dim, out_dim in zip(fc[:-1], fc[1:]):
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.LeakyReLU(0.2, inplace=True))
        fc_layers.append(nn.Linear(fc[-1], 2))
        self.fc_layers = nn.Sequential(*fc_layers)
        self.criterion = nn.CrossEntropyLoss()

    def losses(self, pts_feats, img_feats, attract=True):
        ret = []
        if isinstance(img_feats, torch.Tensor):
            B, C, H, W = img_feats.shape
            img_avg_feats = torch.mean(img_feats.reshape(B, C, -1), dim=-1)
            if attract:
                labels = torch.ones([1, ], dtype=torch.long).to(img_feats.device)
            else:
                labels = torch.zeros([1, ], dtype=torch.long).to(img_feats.device)
            for i, pts in enumerate(pts_feats):
                if len(pts) >= self.min_pts_threshold:
                    pts_avg = torch.mean(pts, dim=0)  # (C_pts, )
                    img_avg = img_avg_feats[i]  # (C_img, )
                    cat_avg = torch.cat([img_avg, pts_avg]).unsqueeze(0)
                    logits = self.fc_layers(cat_avg)  # (1, 2)
                    ret.append(self.criterion(logits, labels))
        return ret







