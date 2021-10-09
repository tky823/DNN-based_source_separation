import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

EPS = 1e-12

class TDF2d(nn.Module):
    """
    Time-Distributed Fully-connected Layer for time-frequency representation
    """
    def __init__(self, num_features, in_bins, out_bins, bias=False, nonlinear='relu', eps=EPS):
        super().__init__()

        self.net = TransformBlock2d(num_features, in_bins, out_bins, bias=bias, nonlinear=nonlinear, eps=eps)

    def forward(self, input):
        output = self.net(input)

        return output

class MultiheadTDF2d(nn.Module):
    def __init__(self, num_features, in_bins, out_bins, num_heads, bias=False, nonlinear='relu', stack_dim=1, eps=EPS):
        super().__init__()

        self.num_heads = num_heads
        self.stack_dim = stack_dim

        net = []

        for idx in range(num_heads):
            net.append(TransformBlock2d(num_features, in_bins, out_bins, bias=bias, nonlinear=nonlinear, eps=eps))

        self.net = nn.ModuleList(net)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, in_channels, n_bins, n_frames) if stack_dim=1
        Returns:
            output <torch.Tensor>: (batch_size, num_heads, in_channels, n_bins, n_frames) if stack_dim=1
        """
        output = []

        for idx in range(self.num_heads):
            x = self.net[idx](input)
            output.append(x)
        
        output = torch.stack(output, dim=self.stack_dim)

        return output

class TransformBlock2d(nn.Module):
    def __init__(self, num_features, in_bins, out_bins, bias=False, nonlinear='relu', eps=EPS):
        super().__init__()

        self.nonlinear = nonlinear

        self.conv1d = nn.Conv1d(in_bins, out_bins, kernel_size=1, stride=1, bias=bias)
        self.norm2d = nn.BatchNorm2d(num_features, eps=eps)

        if nonlinear:
            if nonlinear == 'relu':
                self.nonlinear2d = nn.ReLU()
            else:
                raise ValueError("Not support nonlinear {}".format(nonlinear))
    
    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, num_features, in_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, num_features, out_bins, n_frames)
        """
        batch_size, num_features, _, n_frames = input.size()

        x = input.view(batch_size * num_features, -1, n_frames)
        x = self.conv1d(x)
        x = x.view(batch_size, num_features, -1, n_frames)
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x
        
        return output

class TFC2d(nn.Module):
    """
    Time-Frequency Convolutions
    """
    def __init__(self, in_channels, growth_rate, kernel_size, num_layers=2, nonlinear='relu'):
        super().__init__()

        _in_channels = in_channels

        net = []
        
        for idx in range(num_layers):
            net.append(TFCTransformBlock2d(_in_channels, growth_rate=growth_rate, kernel_size=kernel_size, stride=(1,1), nonlinear=nonlinear))
            _in_channels += growth_rate

        self.net = nn.Sequential(*net)

        self.num_layers = num_layers

    def forward(self, input):
        stack = input
        for idx in range(self.num_layers):
            x = self.net[idx](stack)
            if idx == self.num_layers - 1:
                output = x
            else:
                stack = torch.cat([stack, x], dim=1)

        return output

class TFCTransformBlock2d(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, stride=(1,1), nonlinear='relu', eps=EPS):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.nonlinear = nonlinear

        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=kernel_size, stride=stride)
        self.norm2d = nn.BatchNorm2d(growth_rate, eps=eps)

        if nonlinear:
            if nonlinear == 'relu':
                self.nonlinear2d = nn.ReLU()
            else:
                raise ValueError("Not support nonlinear {}".format(nonlinear))
    
    def forward(self, input):
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride

        padding_height = Kh - Sh
        padding_width = Kw - Sw
        padding_top = padding_height // 2
        padding_left = padding_width // 2
        padding_bottom = padding_height - padding_top
        padding_right = padding_width - padding_left
        
        x = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.conv2d(x)
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x

        return output

def _test_tfc():
    batch_size = 4
    n_bins, n_frames = 257, 128
    in_channels, growth_rate = 2, 3
    kernel_size = (2, 4)

    input = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float)
    model = TFC2d(in_channels, growth_rate=growth_rate, kernel_size=kernel_size)

    output = model(input)

    print(input.size(), output.size())
    

if __name__ == '__main__':
    torch.manual_seed(111)

    _test_tfc()