import torch
import torch.nn as nn

import ai8x

class AI85Net_Fruit(nn.Module):
    def __init__(self, num_classes=7, num_channels=3,dimensions=(64, 64),  bias=False, **kwargs):
        super().__init__()

        self.conv1 = ai8x.FusedMaxPoolConv2dReLU(num_channels, 16, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
        self.conv2 = ai8x.FusedConv2dReLU(16, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv3 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv4 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid1 = ai8x.Add()
        self.conv5 = ai8x.FusedMaxPoolConv2dReLU(20, 20, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
        self.conv6 = ai8x.FusedConv2dReLU(20, 20, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid2 = ai8x.Add()
        self.conv7 = ai8x.FusedConv2dReLU(20, 44, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.conv8 = ai8x.FusedMaxPoolConv2dReLU(44, 48, 3, pool_size=2, pool_stride=2,
                                                 stride=1, padding=1, bias=bias, **kwargs)
        self.conv9 = ai8x.FusedConv2dReLU(48, 48, 3, stride=1, padding=1, bias=bias, **kwargs)
        self.resid3 = ai8x.Add()
        self.conv10 = ai8x.FusedMaxPoolConv2dReLU(48, 32, 3, pool_size=2, pool_stride=2,
                                                  stride=1, padding=0, bias=bias, **kwargs)

        self.fc = ai8x.Linear(32*2*2, num_classes, bias=True, wide=True, **kwargs)
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""

        x = self.conv1(x)  # 16x32x32
        x_res = self.conv2(x)  # 20x32x32
        x = self.conv3(x_res)  # 20x32x32
        x = self.resid1(x, x_res)  # 20x32x32
        x = self.conv4(x)  # 20x32x32
        x_res = self.conv5(x)  # 20x16x16
        x = self.conv6(x_res)  # 20x16x16
        x = self.resid2(x, x_res)  # 20x16x16
        x = self.conv7(x)  # 44x16x16
        x_res = self.conv8(x)  # 48x8x8
        x = self.conv9(x_res)  # 48x8x8
        x = self.resid3(x, x_res)  # 48x8x8
        x = self.conv10(x)  # 96x4x4
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85net_fruit(pretrained=False, **kwargs):
    """
    Constructs a AI85Net5 model.
    """
    assert not pretrained
    return AI85Net_Fruit(**kwargs)

models = [
    {
        'name': 'ai85net_fruit',
        'min_input': 1,
        'dim': 2,
    },
]
