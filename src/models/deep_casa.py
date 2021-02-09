import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_tasnet import choose_layer_norm
from conv import DepthwiseSeparableConv1d

EPS=1e-12

"""
    Temporal Convolutional Network
    See "Temporal Convolutional Networks for Action Segmentation and Detection"
    https://arxiv.org/abs/1611.05267
"""

class TemporalConvNet(nn.Module):
    def __init__(self, num_features, hidden_channels=256, kernel_size=3, num_blocks=3, num_layers=10, dilated=True, separable=False, nonlinear=None, norm=True, causal=True, eps=EPS):
        super().__init__()
        
        self.num_blocks = num_blocks
        
        net = []
        
        for idx in range(num_blocks):    
            net.append(ConvBlock1d(num_features, hidden_channels=hidden_channels, kernel_size=kernel_size, num_layers=num_layers, dilated=dilated, separable=separable, nonlinear=nonlinear, norm=norm, causal=causal, eps=eps))
        
        self.net = nn.Sequential(*net)
        
    def forward(self, input):
        output = self.net(input)
        
        return output


class ConvBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, kernel_size=3, num_layers=10, dilated=True, separable=False, nonlinear=None, norm=True, causal=True, eps=EPS):
        super().__init__()
        
        self.num_layers = num_layers
        
        net = []
        
        for idx in range(num_layers):
            if dilated:
                dilation = 2**idx
                stride = 1
            else:
                dilation = 1
                stride = 2
            net.append(ResidualBlock1d(num_features, hidden_channels=hidden_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, separable=separable, nonlinear=nonlinear, norm=norm, causal=causal, eps=eps))
            
        self.net = nn.Sequential(*net)

    def forward(self, input):
        output = self.net(input)

        return output

        
class ResidualBlock1d(nn.Module):
    def __init__(self, num_features, hidden_channels=256, kernel_size=3, stride=2, dilation=1, separable=False, nonlinear=None, norm=True, causal=True, eps=EPS):
        super().__init__()
        
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.separable, self.causal = separable, causal
        self.norm = norm
        
        self.bottleneck_conv1d_in = nn.Conv1d(num_features, hidden_channels, kernel_size=1, stride=1)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d_in = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
            self.nonlinear = True
        else:
            self.nonlinear = False
        
        if norm:
            self.norm1d_in = choose_layer_norm(hidden_channels, causal, eps=eps)
        
        if separable:
            self.conv1d = DepthwiseSeparableConv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        else:
            self.conv1d = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride, dilation=dilation)
        
        if nonlinear is not None:
            if nonlinear == 'prelu':
                self.nonlinear1d_out = nn.PReLU()
            else:
                raise ValueError("Not support {}".format(nonlinear))
        
        if norm:
            self.norm1d_out = choose_layer_norm(hidden_channels, causal, eps=eps)
        
        self.bottleneck_conv1d_out = nn.Conv1d(hidden_channels, num_features, kernel_size=1, stride=1)
            
    def forward(self, input):
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        nonlinear, norm = self.nonlinear, self.norm
        causal = self.causal
        
        _, _, T_original = input.size()
        
        residual = input
        x = self.bottleneck_conv1d_in(input)
        
        if nonlinear:
            x = self.nonlinear1d_in(x)
        if norm:
            x = self.norm1d_in(x)
        
        padding = (T_original - 1) * stride - T_original + (kernel_size - 1) * dilation + 1
        
        if causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding//2
            padding_right = padding - padding_left

        x = F.pad(x, (padding_left, padding_right))
        x = self.conv1d(x)
        if nonlinear:
            x = self.nonlinear1d_out(x)
        if norm:
            x = self.norm1d_out(x)
        
        x = self.bottleneck_conv1d_out(x)
        output = x + residual
            
        return output

def _test_residual_block():
    batch_size, T = 4, 32
    in_channels, out_channels = 16, 32

    input = torch.randn((batch_size, in_channels, T), dtype=torch.float)

    print('-'*10, 'Causal', '-'*10)
    kernel_size, stride = 3, 1
    dilation = 2
    causal = True

    model = ResidualBlock1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal)
    print(model)
    output = model(input)
    print(input.size(), output.size())
    print()

    print('-'*10, 'Non causal', '-'*10)
    kernel_size, stride = 3, 2
    dilation = 1
    separable = True
    causal = False
    
    model = ResidualBlock1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, causal=causal, separable=separable)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_conv_block():
    batch_size, C, T = 4, 3, 32

    input = torch.randn((batch_size, C, T), dtype=torch.float)

    model = ConvBlock1d(C, hidden_channels=32, kernel_size=3, num_layers=4, dilated=True, separable=True)
    print(model)
    output = model(input)
    print(input.size(), output.size())

def _test_tcn():
    batch_size = 4
    T = 128
    in_channels, hidden_channels = 16, 32
    kernel_size = 3
    num_blocks = 3
    num_layers = 4
    dilated, separable = True, True
    causal = True
    nonlinear = 'prelu'
    norm = True
    
    input = torch.randn((batch_size, in_channels, T), dtype=torch.float)
    
    model = TemporalConvNet(
        in_channels, hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_blocks=num_blocks, num_layers=num_layers,
        dilated=dilated, separable=separable,
        nonlinear=nonlinear, norm=norm,
        causal=causal
    )
    print(model)
    output = model(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print('='*10, 'ResidualBlock1d', '='*10)
    _test_residual_block()
    print()

    print('='*10, 'ConvBlock1d', '='*10)
    _test_conv_block()
    print()

    print('='*10, 'TCN', '='*10)
    _test_tcn()
