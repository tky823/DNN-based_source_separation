import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import MultiDilatedConv2d

EPS=1e-12

class D2Block(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, norm=True, nonlinear='relu', eps=EPS):
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
            net.append(MultiDilatedConvBlock(num_features, growth_rate[idx], kernel_size=kernel_size, norm=norm, nonlinear=nonlinear, eps=eps))
        self.net = nn.Sequential(*net)

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output 
                (batch_size, num_blocks * sum(growth_rate), n_bins, n_frames) if type(growth_rate) is list<int>
                or (batch_size, num_blocks * depth * growth_rate, n_bins, n_frames) if type(growth_rate) is int
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
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, nonlinear='relu', eps=EPS):
        super().__init__()

        assert type(in_channels) is list, "`in_channels` must be type of list."

        self.norm = norm
        self.nonlinear = nonlinear

        if self.norm:
            self.norm2d = nn.BatchNorm2d(sum(in_channels), eps=eps)
        if self.nonlinear:
            if self.nonlinear == 'relu':
                self.nonlinear2d = nn.ReLU()
            else:
                raise NotImplementedError("Not support nonlinearity {}".format(self.nonlinear))
        self.conv2d = MultiDilatedConv2d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, input):
        """
        Args:
            input (batch_size, sum(in_channels), n_bins, n_frames)
            output (batch_size, out_channels, n_bins, n_frames)
        """
        x = input
        if self.norm:
            x = self.norm2d(x)
        if self.nonlinear:
            x = self.nonlinear2d(x)
        output = self.conv2d(x)

        return output

def _test_multidilated_conv_block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, growth_rate = [3, 4, 8], 2
    kernel_size = (3, 3)

    input = torch.randn(batch_size, sum(in_channels), H, W)
    model = MultiDilatedConvBlock(in_channels, growth_rate, kernel_size=kernel_size)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d2block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels, growth_rate = 4, 2
    depth = 4

    input = torch.randn(batch_size, in_channels, H, W)
    model = D2Block(in_channels, growth_rate, kernel_size=(3,3), depth=depth)
    print(model)
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    _test_multidilated_conv_block()
    print()

    _test_d2block()
    print()