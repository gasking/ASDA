import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self,
                 inc,
                 num_classes
                 ):
        super(LeNet, self).__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels = inc, out_channels = 6, kernel_size = 5, stride = 1),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace = True)

        )

        self.s2 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.c3 = nn.Sequential(
            nn.Conv2d(in_channels = 6,
                      out_channels = 16,
                      stride = 1, kernel_size = 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True)
        )

        self.s4 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.c5 = nn.Sequential(
            nn.Conv2d(in_channels = 16,
                      out_channels = 16,
                      stride = 1, kernel_size = 5, padding = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True)
        )

        # --------------------------------------------#
        #               上采样
        # --------------------------------------------#
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 16,
                               out_channels = 32,
                               stride = 2,
                               kernel_size = 3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = 32,
                               out_channels = 32,
                               stride = 1,
                               kernel_size = 6),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),

            nn.Upsample(scale_factor = 2),
            nn.Conv2d(in_channels = 32, out_channels = 32,
                      kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 32, out_channels = num_classes,
                      kernel_size = 1)

        )

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  # 冻结BN层
    def forward(self, x):
        x = x.float()
        x = self.c1(x)
        x = self.s2(x)
        x = self.c3(x)
        x = self.s4(x)
        x = self.c5(x)  # 三维向量

        # ---------------------#
        #        分割
        # ---------------------#
        seg = self.up(x)

        return seg


if __name__ == '__main__':
    x = torch.randn((2, 3, 300, 300))

    model = LeNet(inc = 3, num_classes = 10)

    seg = model.forward(x)

    print(seg.shape)
