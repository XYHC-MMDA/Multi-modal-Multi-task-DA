import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class InfoNCE(nn.Module):
    def __init__(self, temperature=1.0, contrast_mode='cross_entropy', reduction='mean'):
        super(InfoNCE, self).__init__()
        self.contrast_mode = contrast_mode
        self.T = temperature

    def forward(self, pts_feats, img_feats):
        '''
            features: (N, 64)
        '''
        num_pts = len(pts_feats)
        pts_norm = F.normalize(pts_feats, dim=1)
        img_norm = F.normalize(img_feats, dim=1)
        logits = torch.matmul(pts_norm, img_norm.T) / self.T  # (num_pts, num_pts)
        if self.contrast_mode == 'cross_entropy':
            labels = torch.arange(num_pts).to(pts_feats.device)
            contrast_loss = F.cross_entropy(logits, labels)
        else:
            logits_max = torch.max(logits, dim=1)[0].unsqueeze(1)
            logits_stable = logits - logits_max
            mask = torch.eye(num_pts, dtype=torch.bool)
            positives = logits_stable[mask]
            negatives = logits_stable[~mask].view(num_pts, -1)
            neg_exp = torch.exp(negatives)
            pos_logsoftmax = positives - torch.log(torch.sum(neg_exp, dim=1))
            contrast_loss = - torch.mean(pos_logsoftmax)
        return contrast_loss



