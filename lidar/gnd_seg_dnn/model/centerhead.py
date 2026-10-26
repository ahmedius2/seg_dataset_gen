import torch
import torch.nn as nn
from functools import partial

def hm_conv_block(input_channels, out_channels, num_conv, init_bias=-2.19, use_bias=False, norm_func=None):
    fc_list = []
    for k in range(num_conv - 1):
        fc_list.append(nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(input_channels) if norm_func is None else norm_func(input_channels),
            nn.ReLU()
        ))
    fc_list.append(nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True))
    fc_list[-1].bias.data.fill_(init_bias)
    return fc_list

class CenterHead(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        norm_func = partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, 3, stride=1, padding=1,
                bias=True
            ),
            norm_func(64),
            nn.ReLU(),
            *hm_conv_block(
                input_channels=64,
                out_channels=1,
                num_conv=2,
                init_bias=-2.19,
                use_bias=True,
                norm_func=norm_func
            )
        )

    #def sigmoid(self, x):
    #    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    #    return y

    def forward(self, x):
        return self.layers(x)