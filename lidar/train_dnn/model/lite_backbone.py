"""Small BEV network built only from ops TIDL (TI C7x) handles well:
3x3/1x1 Conv, BN, ReLU, Concat, nearest/bilinear Resize. All heavy work runs
at <= 1/2 input resolution (~2 GMAC vs ~30 GMAC for the full model)."""

import torch
import torch.nn as nn


def _cbr(cin, cout, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(cout, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class LiteBEVNet(nn.Module):
    def __init__(self, in_ch=32, chans=(32, 64, 128), mid=64, head_ch=32):
        super().__init__()
        c1, c2, c3 = chans
        self.s1 = nn.Sequential(_cbr(in_ch, c1, s=2), _cbr(c1, c1))
        self.s2 = nn.Sequential(_cbr(c1, c2, s=2), _cbr(c2, c2), _cbr(c2, c2))
        self.s3 = nn.Sequential(_cbr(c2, c3, s=2), _cbr(c3, c3), _cbr(c3, c3))
        # 1x1 at low resolution first, then resize: much cheaper than ConvTranspose
        self.l1 = _cbr(c1, mid, k=1)
        self.l2 = _cbr(c2, mid, k=1)
        self.l3 = _cbr(c3, mid, k=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4 = nn.Upsample(scale_factor=4, mode="nearest")
        self.fuse = _cbr(mid * 3, head_ch)
        self.out = nn.Conv2d(head_ch, 1, 1)
        self.out.bias.data.fill_(-2.19)
        self.up_out = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        y = torch.cat([self.l1(f1), self.up2(self.l2(f2)), self.up4(self.l3(f3))], dim=1)
        return self.up_out(self.out(self.fuse(y)))
