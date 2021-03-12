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
    def __init__(self, eps=EPS, **kwargs):
        super().__init__()

        self.eps = eps

        raise NotImplementedError("Implement D3Net")
    
    def forward(self, input):
        raise NotImplementedError("Implement D3Net")

class D3Block(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size, dilations=[(1,1),(2,2),(4,4)], num_blocks=3, eps=EPS):
        super().__init__()

        self.num_blocks = num_blocks

        net = []
        for idx in range(num_blocks):
            net.append(D2Block(in_channels, num_features, kernel_size, dilations=dilations, eps=eps))
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

        for idx in range(self.num_blocks):
            stacked.append(x)
            x = torch.cat(stacked, dim=1)
            x = self.net[idx](x)
            
        output = x

        return output

class D2Block(nn.Module):
    def __init__(self, in_channels, num_features, kernel_size, dilations=[(1,1),(2,2),(4,4)], eps=EPS):
        super().__init__()

        self.num_dilations = len(dilations)

        net = []
        stacked_in_channels = []

        for idx in range(self.num_dilations):
            if len(stacked_in_channels) == 0:
                stacked_in_channels.append(in_channels)
                net.append(MultiDilatedConv2d(stacked_in_channels, num_features, kernel_size=kernel_size, dilations=dilations[:idx+1]))
            else:
                stacked_in_channels.append(num_features)
                net.append(MultiDilatedConv2d(stacked_in_channels, num_features, kernel_size=kernel_size, dilations=dilations[:idx+1]))
                num_features += num_features
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

        for idx in range(self.num_dilations):
            stacked.append(x)
            x = self.net[idx](*stacked)
            
        output = x

        return output

def _test_dense_block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels = 3
    num_features = 4
    dilations = [(1,1),(2,2),(4,4)]

    input = torch.randn(batch_size, in_channels, H, W)

    model = D2Block(in_channels, num_features, kernel_size=(3,3), dilations=dilations)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_d3block():
    torch.manual_seed(111)
    
    batch_size = 4
    H, W = 16, 32
    in_channels = 3
    num_features = 4
    dilations = [(1,1),(2,2),(4,4)]

    input = torch.randn(batch_size, in_channels, H, W)

    model = D3Block(in_channels, num_features, kernel_size=(3,3), dilations=dilations)
    print(model)
    output = model(input)
    print(input.size(), output.size())



if __name__ == '__main__':
    # _test_multi_dilated_conv()
    # _test_dense_block()
    _test_d3block()