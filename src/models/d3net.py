import torch
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F

from conv import MultiDilatedConv2d

"""
Reference: D3Net: Densely connected multidilated DenseNet for music source separation
See https://arxiv.org/abs/2010.01733
"""

EPS=1e-12

class D3Net(nn.Module):
    def __init__(self, eps=EPS, **kwargs):
        super().__init__()

        self.eps = eps

        raise NotImplementedError("Implement D3Net")
    
    def forward(self, input):
        raise NotImplementedError("Implement D3Net")

class D3Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_blocks=3, depth=None, eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks
        if type(out_channels) is int:
            out_channels = [
                out_channels for _ in range(num_blocks)
            ]
            if depth is None or type(depth) is int:
                depth = [
                    None for _ in range(num_blocks)
                ]
        elif type(out_channels) is list:
            if depth is None or type(depth) is int:
                depth = [
                    None for _ in range(num_blocks)
                ]
        else:
            raise ValueError("Not support `out_channels`={}".format(out_channels))

        net = []
        for idx in range(num_blocks):
            net.append(D2Block(in_channels, out_channels, kernel_size, depth=depth[idx], eps=eps))
            in_channels += out_channels[-1]
        
        self.net = nn.Sequential(*net)

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output (batch_size, out_channels, n_bins, n_frames)
        """
        x = input
        stacked = None

        for idx in range(self.num_blocks):
            if stacked is None:
                stacked = x
            else:
                stacked = torch.cat([stacked, x], dim=1)
            x = self.net[idx](stacked)
            
        output = x

        return output

class D2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth=None, eps=EPS):
        super().__init__()

        if type(out_channels) is int:
            assert depth is not None, "Specify `depth`"
            out_channels = [
                out_channels for _ in range(depth)
            ]
        elif type(out_channels) is list:
            depth = len(out_channels)
        else:
            raise ValueError("Not support out_channels={}".format(out_channels))
        self.depth = depth

        net = []
        num_features = []

        for idx in range(self.depth):
            if idx == 0:
                num_features.append(in_channels)
            else:
                num_features.append(out_channels[idx-1])
            net.append(MultiDilatedConv2d(num_features, out_channels[idx], kernel_size=kernel_size))
        self.net = nn.Sequential(*net)

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output (batch_size, out_channels, n_bins, n_frames)
        """
        x = input
        stacked = None

        for idx in range(self.depth):
            if stacked is None:
                stacked = x
            else:
                stacked = torch.cat([stacked, x], dim=1)
            x = self.net[idx](stacked)
        
        output = x

        return output

class MultiDilatedConvBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size):
        super().__init__()

        self.conv2d = MultiDilatedConv2d(in_channels, growth_rate, kernel_size=kernel_size)

    def forward(self, input):
        x = self.conv2d(input)
        output = torch.cat([input, x], dim=1)

        return output

def _test_d2block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, out_channels = 3, [2, 4, 6]

    input = torch.randn(batch_size, in_channels, H, W)
    model = D2Block(in_channels, out_channels, kernel_size=(3,3))
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, out_channels = 3, [2, 4, 6]
    num_blocks = 3

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3Block(in_channels, out_channels, kernel_size=(3,3), num_blocks=num_blocks)
    print(model)
    output = model(input)
    print(input.size(), output.size())



if __name__ == '__main__':
    # _test_d2block()
    _test_d3block()