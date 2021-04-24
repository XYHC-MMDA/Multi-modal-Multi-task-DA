import torch
from mmdet3d.models import UNetSCN, UNetSCNManual

def test():
    b, n, DIMENSION = 1, 10, 3
    coords = torch.randint(4096, [b, n, DIMENSION])
    batch_idxs = torch.arange(b).reshape(b, 1, 1).repeat(1, n, 1)
    coords = torch.cat([coords, batch_idxs], 2).reshape(-1, DIMENSION + 1)

    in_channels, out_channels = 1, 5
    feats = torch.rand(b * n, in_channels)

    x = [coords, feats]

    model1 = UNetSCN(in_channels, m=out_channels)
    model2 = UNetSCNManual(in_channels, m=out_channels)
    out1 = model1(x)
    out2 = model2(x)
    f = open('./dedebug/scn_manual.txt', 'w')
    f.write(str(model2))
    f.close()
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    test()
