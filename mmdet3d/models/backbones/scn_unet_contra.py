import torch
import torch.nn as nn
import sparseconvnet as scn
from mmdet.models import BACKBONES


@BACKBONES.register_module()
class UNetSCNContra(nn.Module):
    def __init__(self,
                 in_channels,
                 m=16,  # number of unet features (multiplied in each layer)
                 block_reps=1,  # depth
                 residual_blocks=False,  # ResNet style basic blocks
                 full_scale=4096,
                 num_planes=7,
                 DIMENSION=3,
                 downsample=[2, 2],
                 leakiness=0,
                 n_input_planes=-1,
                 lout=5,
                 ):
        super(UNetSCNContra, self).__init__()

        # parameters
        self.in_channels = in_channels
        self.out_channels = m
        self.block_reps = block_reps
        self.residual_blocks = residual_blocks
        self.full_scale = full_scale
        self.n_planes = [(n + 1) * m for n in range(num_planes)]
        self.dimension = DIMENSION
        self.downsample = downsample
        self.leakiness = leakiness
        self.lout = lout
        # self.n_input_planes = n_input_planes

        # before U-Net
        self.input_layer = scn.InputLayer(DIMENSION, full_scale, mode=4)
        self.SC1 = scn.SubmanifoldConvolution(DIMENSION, in_channels, self.out_channels, 3, False)

        # U-Net
        self.enc_convs, self.middle_conv, self.dec_convs = self.iter_unet(n_input_planes)

        # after U-Net
        self.BNReLU = scn.BatchNormReLU(self.out_channels)
        self.inter_out_layer = scn.OutputLayer(self.dimension)
        self.output_layer = scn.OutputLayer(self.dimension)

    def block(self, n_in, n_out):
        m = scn.Sequential()
        if self.residual_blocks:  # ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if n_in == n_out else scn.NetworkInNetwork(n_in, n_out, False))
                  .add(scn.Sequential()
                       .add(scn.BatchNormLeakyReLU(n_in, leakiness=self.leakiness))
                       .add(scn.SubmanifoldConvolution(self.dimension, n_in, n_out, 3, False))
                       .add(scn.BatchNormLeakyReLU(n_out, leakiness=self.leakiness))
                       .add(scn.SubmanifoldConvolution(self.dimension, n_out, n_out, 3, False)))
                  )
            m.add(scn.AddTable())
        else:  # VGG style blocks
            m.add(scn.BatchNormLeakyReLU(n_in, leakiness=self.leakiness))
            m.add(scn.SubmanifoldConvolution(self.dimension, n_in, n_out, 3, False))
        return m

    def iter_unet(self, n_input_planes):
        # different from scn implementation, which is a recursive function
        enc_convs = scn.Sequential()
        dec_convs = scn.Sequential()
        for n_planes_in, n_planes_out in zip(self.n_planes[:-1], self.n_planes[1:]):
            # encode
            conv1x1 = scn.Sequential()
            for i in range(self.block_reps):
                conv1x1.add(self.block(n_input_planes if n_input_planes != -1 else n_planes_in, n_planes_in))
                n_input_planes = -1

            conv = scn.Sequential()
            conv.add(scn.BatchNormLeakyReLU(n_planes_in, leakiness=self.leakiness))
            conv.add(scn.Convolution(self.dimension, n_planes_in, n_planes_out,
                                     self.downsample[0], self.downsample[1], False))
            enc_conv = scn.Sequential()
            enc_conv.add(conv1x1)
            enc_conv.add(conv)
            enc_convs.add(enc_conv)

            # decode(corresponding stage of encode; symmetric with U)
            b_join = scn.Sequential()  # before_join
            b_join.add(scn.BatchNormLeakyReLU(n_planes_out, leakiness=self.leakiness))
            b_join.add(scn.Deconvolution(self.dimension, n_planes_out, n_planes_in,
                                         self.downsample[0], self.downsample[1], False))
            join_table = scn.JoinTable()
            a_join = scn.Sequential()  # after_join
            for i in range(self.block_reps):
                a_join.add(self.block(n_planes_in * (2 if i == 0 else 1), n_planes_in))
            dec_conv = scn.Sequential()
            dec_conv.add(b_join)
            dec_conv.add(join_table)
            dec_conv.add(a_join)
            dec_convs.add(dec_conv)

        middle_conv = scn.Sequential()
        for i in range(self.block_reps):
            middle_conv.add(self.block(n_input_planes if n_input_planes != -1 else self.n_planes[-1], self.n_planes[-1]))
            n_input_planes = -1

        return enc_convs, middle_conv, dec_convs

    def forward(self, x):
        # before U-Net
        x = self.input_layer(x)
        x = self.SC1(x)

        # U-Net
        inter_features = []
        for convs in self.enc_convs:
            conv1x1, conv = convs[0], convs[1]
            x = conv1x1(x)
            inter_features.append(x)
            x = conv(x)
        assert len(inter_features) == len(self.dec_convs)

        x = self.middle_conv(x)

        x_m = None
        for i in reversed(range(len(inter_features))):
            inter_feat = inter_features[i]
            deconvs = self.dec_convs[i]
            b_join, join_table, a_join = deconvs[0], deconvs[1], deconvs[2]

            x = b_join(x)
            x = join_table([inter_feat, x])
            x = a_join(x)

            if len(inter_features) - i == self.lout:
                x_m = self.inter_out_layer(x)

        # after U-Net
        x = self.BNReLU(x)
        x = self.output_layer(x)

        return x_m, x


def test():
    # see dedebug/scn_manual_test.py
    b, n, DIMENSION = 1, 10, 3
    coords = torch.randint(4096, [b, n, DIMENSION])
    batch_idxs = torch.arange(b).reshape(b, 1, 1).repeat(1, n, 1)
    coords = torch.cat([coords, batch_idxs], 2).reshape(-1, DIMENSION + 1)

    in_channels, out_channels = 1, 5
    feats = torch.rand(b * n, in_channels)

    x = [coords, feats]

    from mmdet3d.models import UNetSCN
    model1 = UNetSCN(in_channels, m=out_channels)
    model2 = UNetSCNContra(in_channels, m=out_channels)
    out1 = model1(x)
    out2 = model2(x)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    test()
