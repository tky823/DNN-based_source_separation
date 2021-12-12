import torch.nn as nn
import torch.nn.functional as F

from utils.model import choose_nonlinear

EPS = 1e-12

class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, kernel_size=(3,3), nonlinear='relu', eps=EPS):
        super().__init__()

        self.kernel_size = kernel_size

        self.bottleneck_conv2d_in = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bottleneck_norm2d_in = nn.BatchNorm2d(bottleneck_channels, eps=eps)
        self.bottleneck_nonlinear2d_in = choose_nonlinear(nonlinear)

        self.conv2d = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel_size, stride=(1,1), bias=False)
        self.norm2d = nn.BatchNorm2d(bottleneck_channels, eps=eps)
        self.nonlinear2d = choose_nonlinear(nonlinear)

        self.bottleneck_conv2d_out = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bottleneck_norm2d_out = nn.BatchNorm2d(out_channels, eps=eps)

        if out_channels != in_channels:
            self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.pointwise_conv2d = None

        self.bottleneck_nonlinear2d_out = choose_nonlinear(nonlinear)
    
    def forward(self, input):
        Kh, Kw = self.kernel_size
        padding_height, padding_width = Kh - 1, Kw - 1
        padding_top, padding_left = padding_height // 2, padding_width // 2
        padding_bottom, padding_right = padding_height - padding_top, padding_width - padding_left

        x = self.bottleneck_conv2d_in(input)
        x = self.bottleneck_norm2d_in(x)
        x = self.bottleneck_nonlinear2d_in(x)
        x = F.pad(x, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.conv2d(x)
        x = self.norm2d(x)
        x = self.nonlinear2d(x)
        x = self.bottleneck_conv2d_out(x)
        x = self.bottleneck_norm2d_out(x)

        if self.pointwise_conv2d:
            x_residual = self.pointwise_conv2d(input)
        else:
            x_residual = input

        x = x + x_residual
        output = self.nonlinear2d(x)

        return output

def _test_residual_block():
    batch_size = 4
    H, W = 16, 20
    in_channels, out_channels, hidden_channels = 8, 8, 4

    input = torch.randn(batch_size, in_channels, H, W)
    model = ResidualBlock2d(in_channels, out_channels, hidden_channels)
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    torch.manual_seed(111)

    print("="*10, "ResidualBlock", "="*10)
    _test_residual_block()
    print()