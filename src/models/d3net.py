import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import MultiDilatedConv2d

"""
Reference: D3Net: Densely connected multidilated DenseNet for music source separation
See https://arxiv.org/abs/2010.01733
"""

EPS=1e-12

class D3Net(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, num_d3blocks=4, num_d2blocks=3, depth=None, eps=EPS, **kwargs):
        super().__init__()

        if type(growth_rate) is int:
            growth_rate = [
                growth_rate for _ in range(num_d3blocks)
            ]
        elif type(growth_rate) is list:
            pass
        else:
            raise ValueError("Not support `growth_rate`={}".format(growth_rate))

        if type(num_d2blocks) is int:
            num_d2blocks = [
                num_d2blocks for _ in range(num_d3blocks)
            ]
        elif type(num_d2blocks) is list:
            pass
        else:
            raise ValueError("Not support `num_d2blocks`={}".format(num_d2blocks))

        if depth is None:
            depth = [
                None for _ in range(num_d3blocks)
            ]
        elif type(depth) is int:
            depth = [
                depth for _ in range(num_d3blocks)
            ]

        net = []

        for idx in range(num_d3blocks):
            net.append(D3Block(in_channels, growth_rate[idx], kernel_size, num_blocks=num_d2blocks[idx], depth=depth[idx], eps=eps))
            in_channels += growth_rate[idx] * num_d2blocks[idx] * depth[idx]
        
        self.net = nn.Sequential(*net)

        self.num_d3blocks = num_d3blocks

        self.eps = eps

    
    def forward(self, input):
        x = input
        stacked = []

        stacked.append(input)

        for idx in range(self.num_d3blocks):
            if idx != 0:
                x = torch.cat(stacked, dim=1)
            x = self.net[idx](x)
            stacked.append(x)
        
        output = torch.cat(stacked[1:], dim=1)

        return output

class D3Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, num_blocks=3, depth=None, eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks

        if type(growth_rate) is int:
            growth_rate = [
                growth_rate for _ in range(num_blocks)
            ]
        elif type(growth_rate) is list:
            pass
        else:
            raise ValueError("Not support `growth_rate`={}".format(growth_rate))
            
        if depth is None:
            depth = [
                None for _ in range(num_blocks)
            ]
        elif type(depth) is int:
            depth = [
                depth for _ in range(num_blocks)
            ]

        net = []

        for idx in range(num_blocks):
            net.append(D2Block(in_channels, growth_rate[idx], kernel_size, depth=depth[idx], eps=eps))
            in_channels += growth_rate[idx] * depth[idx]
        
        self.net = nn.Sequential(*net)

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output (batch_size, out_channels, n_bins, n_frames)
        """
        x = input
        stacked = []

        stacked.append(input)

        for idx in range(self.num_blocks):
            if idx != 0:
                x = torch.cat(stacked, dim=1)
            x = self.net[idx](x)
            stacked.append(x)
        
        output = torch.cat(stacked[1:], dim=1)

        return output

class D2Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, eps=EPS):
        super().__init__()

        if type(growth_rate) is int:
            assert depth is not None, "Specify `depth`"
            growth_rate = [
                growth_rate for _ in range(depth)
            ]
        elif type(growth_rate) is list:
            if depth is not None:
                assert depth == len(growth_rate), "`depth` is different from `len(growth_rate)`"
            depth = len(growth_rate)
        else:
            raise ValueError("Not support growth_rate={}".format(growth_rate))
        self.depth = depth

        net = []
        num_features = []

        for idx in range(self.depth):
            if idx == 0:
                num_features.append(in_channels)
            else:
                num_features.append(growth_rate[idx-1])
            net.append(MultiDilatedConv2d(num_features, growth_rate[idx], kernel_size=kernel_size))
        self.net = nn.Sequential(*net)

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output (batch_size, out_channels, n_bins, n_frames)
        """
        x = input
        stacked = []
        output = []

        stacked.append(input)

        for idx in range(self.depth):
            if idx != 0:
                x = torch.cat(stacked, dim=1)
            x = self.net[idx](x)
            stacked.append(x)
        
        output = torch.cat(stacked[1:], dim=1)

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
    in_channels, groth_rate = 4, 2

    input = torch.randn(batch_size, in_channels, H, W)
    model = D2Block(in_channels, groth_rate, kernel_size=(3,3), depth=4)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, growth_rate = 3, [2, 4, 6]
    depth = 4
    num_blocks = 3

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3Block(in_channels, growth_rate, kernel_size=(3,3), num_blocks=num_blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3net():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, growth_rate = 3, [2, 4, 6]
    depth = [4, 5, 6]
    num_d3blocks, num_d2blocks = 3, [2, 3, 4]

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3Net(in_channels, growth_rate, kernel_size=(3,3), num_d3blocks=num_d3blocks, num_d2blocks=num_d2blocks, depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())



if __name__ == '__main__':
    # _test_d2block()
    # _test_d3block()

    _test_d3net()