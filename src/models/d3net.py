import torch
import torch.nn as nn

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

class DenseBlock(nn.Module):
    def __init__(self, num_features, kernel_size, stride=(2,2), eps=EPS, **kwargs):
        super().__init__()

        self.conv

        self.eps = eps
    
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            output (batch_size, out_channels, n_bins, n_frames)
        """

        return input

class MultiDilatedConv2d(nn.Module):
    def __init__(self, num_features, kernel_size=(3,3), dilations=[(1,1),(2,2),(4,4)]):
        self.num_dilations = len(dilations)

        for dilation in dilations:
            self.net.append(nn.Conv2d(num_features, 1, kernel_size=kernel_size, stride=(1,1), dilation=dilation))

    def forward(self, *args):
        input = args
        output = []

        for idx in range(self.num_dilations):
            x = self.net[idx](input[idx])
            output.append(x)
        
        output = torch.cat(x, dim=1)

        return output

def _test_multi_dilated_conv():
    torch.manual_seed(111)
    
    model = MultiDilatedConv2d()

def _test_dense_block():
    model = DenseBlock()



if __name__ == '__main__':
    _test_dense_block()