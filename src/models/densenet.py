import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Reference: 
See 
"""

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride=(1,1), hidden_channels=128):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride

        self.bottleneck_conv2d = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1,1), stride=(1,1))
        self.conv2d = nn.Conv2d(hidden_channels, growth_rate, kernel_size=kernel_size, stride=stride)

    def forward(self, input):
        _, _, H_in, W_in = input.size()

        padding_height = H_in * (self.stride[0] - 1) + self.kernel_size[0]  - self.stride[0]
        padding_width = W_in * (self.stride[1] - 1) + self.kernel_size[1]  - self.stride[1]
        padding_up = padding_height // 2
        padding_bottom = padding_height - padding_up
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left

        x = self.bottleneck_conv2d(input)
        x = F.pad(x, (padding_left, padding_right, padding_up, padding_bottom))
        x = self.conv2d(x)
        output = torch.cat([input, x], dim=1)

        return output

def _test_dense_block():
    torch.manual_seed(111)

    batch_size = 4
    H, W = 16, 32
    in_channels, growth_rate = 32, 16

    input = torch.randn(batch_size, in_channels, H, W)

    model = DenseBlock(in_channels, kernel_size=(3, 5), growth_rate=growth_rate)
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    _test_dense_block()