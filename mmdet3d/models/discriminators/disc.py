import torch.nn as nn
import torch
import torch.nn.functional as F
from ..registry import DISCRIMINATORS


@DISCRIMINATORS.register_module()
class FCDiscriminatorCE(nn.Module):
    def __init__(self, in_dim=128):
        super(FCDiscriminatorCE, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x.shape=(N, in_dim=128)
        x = self.fc(x)
        return x

    def loss(self, logits, src=True):
        if src:
            labels = torch.ones(logits.size(0), dtype=torch.long).cuda()
        else:
            labels = torch.zeros(logits.size(0), dtype=torch.long).cuda()
        return self.criterion(logits, labels)


@DISCRIMINATORS.register_module()
class Conv2dDiscriminator(nn.Module):
    def __init__(self, in_channels=128):
        super(Conv2dDiscriminator, self).__init__()
        dim1, dim2 = 64, 64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim1, kernel_size=3, stride=2),
            # nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, kernel_size=3, stride=2),
            # nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, 2, kernel_size=3, padding=1),
        )
        self.criterion = nn.CrossEntropyLoss()

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
        return self.criterion(logits, labels)


@DISCRIMINATORS.register_module()
class FCDiscriminator(nn.Module):
    def __init__(self, in_dim=128):
        super(FCDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
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
class FCDiscriminatorNew(nn.Module):
    def __init__(self, in_dim=128):
        super(FCDiscriminatorNew, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.nllloss = nn.NLLLoss()

    def forward(self, x, return_logits=False):
        # x.shape=(N, in_dim=128)
        logits = self.fc(x)
        logprob = F.log_softmax(logits, dim=1)
        if return_logits:
            return logits, logprob
        else:
            return logprob

    def loss(self, logprob, src=True):
        if src:
            labels = torch.ones(logprob.size(0), dtype=torch.long).cuda()
        else:
            labels = torch.zeros(logprob.size(0), dtype=torch.long).cuda()
        return self.nllloss(logprob, labels)


if __name__ == '__main__':
    disc = Conv2dDiscriminator(128)
    x = torch.rand(2, 128, 200, 400)
    x = disc(x)
    print(x.shape)  # (2, 2, 49, 99)
