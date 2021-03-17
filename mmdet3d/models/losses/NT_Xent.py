import torch.nn as nn
import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class NT_Xent(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='cross_entropy', normalize='True', reduction='mean'):
        super(NT_Xent, self).__init__()
        self.T = temperature
        self.contrast_mode = contrast_mode
        self.normalize = normalize

    def forward(self, feats_1, feats_2):
        '''
            feats_1.shape == feats_2.shape == (N, C)
        '''
        num_pts = len(feats_1)
        if self.normalize:
            feats_1 = F.normalize(feats_1, dim=1)
            feats_2 = F.normalize(feats_2, dim=1)
        logits = torch.matmul(feats_1, feats_2.T) / self.T  # (num_pts, num_pts)
        if self.contrast_mode == 'cross_entropy':
            labels = torch.arange(num_pts).to(feats_1.device)
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



