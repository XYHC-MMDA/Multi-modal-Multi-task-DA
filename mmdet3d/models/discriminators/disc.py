import torch.nn as nn
import torch
import torch.nn.functional as F
from ..registry import DISCRIMINATORS


@DISCRIMINATORS.register_module()
class SegDiscriminator(nn.Module):
    def __init__(self, in_dim=128):
        super(SegDiscriminator, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 16), nn.Linear(16, 2))
        self.nllloss = nn.NLLLoss()

    def forward(self, x):
        # x.shape=(N, in_dim=128)
        x = self.fc(x)
        return x

    def loss(self, logits, src=True):
        if src:
            labels = torch.ones(logits.size(0), dtype=torch.long).cuda()
        else:
            labels = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        return self.nllloss(F.log_softmax(logits, dim=1), labels)


@DISCRIMINATORS.register_module()
class DetDiscriminator(nn.Module):
    def __init__(self, in_channels=128):
        super(DetDiscriminator, self).__init__()
        dim1, dim2 = 16, 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim1, kernel_size=3, stride=2),
            # nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=2),
            # nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True)
        )
        self.nllloss = nn.NLLLoss()

    def forward(self, x):
        # x.shape=(N, 128, 200, 400)
        x = self.conv(x)  # (N, 2, 49, 99)
        return x

    def loss(self, logits, src=True):
        N, _, H, W = logits.shape
        if src:
            labels = torch.ones([N, H, W], dtype=torch.long).cuda()
        else:
            labels = torch.zeros([N, H, W], dtype=torch.long).cuda()
        return self.nllloss(F.log_softmax(logits, dim=1), labels)


if __name__ == '__main__':
    disc = DetDiscriminator(128)
    x = torch.rand(2, 128, 200, 400)
    x = disc(x)
    print(x.shape)  # (2, 2, 49, 99)
