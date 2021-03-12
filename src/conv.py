import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

"""
    Depthwise Separable Convolution
    See "Xception: Deep Learning with Depthwise Separable Convolutions"
    https://arxiv.org/abs/1610.02357
"""
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=0, dilation=1, bias=True):
        super().__init__()
        
        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.depthwise_conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, input):
        x = self.depthwise_conv1d(input)
        output = self.pointwise_conv1d(x)
        
        return output

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, padding=(0,0), dilation=(1,1), bias=True):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
            
        if stride is None:
            stride = kernel_size

        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=bias)

    def forward(self, input):
        x = self.depthwise_conv2d(input)
        output = self.pointwise_conv2d(x)
        
        return output

        
"""
    Depthwise Separable Transposed Convolution
"""
        
class DepthwiseSeparableConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, bias=True):
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        self.depthwise_conv1d = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels, bias=bias)
        

    def forward(self, input):
        x = self.pointwise_conv1d(input)
        output = self.depthwise_conv1d(input)
        
        return output
        
class DepthwiseSeparableConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=None, dilation=1, bias=True):
        super().__init__()
        
        kernel_size = _pair(kernel_size)
            
        if stride is None:
            stride = kernel_size

        stride = _pair(stride)
        dilation = _pair(dilation)

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        
        self.pointwise_conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1), bias=bias)
        self.depthwise_conv2d = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels, bias=bias)

    def forward(self, input):
        x = self.pointwise_conv2d(input)
        output = self.depthwise_conv2d(x)
        
        return output


"""
    Complex convolution
    See "Deep Complex Networks"
    https://arxiv.org/abs/1705.09792
"""

class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1):
        super().__init__()

        raise NotImplementedError
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.groups = groups
        
        self.weight_real = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        self.weight_imag = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_imag, a=math.sqrt(5))

    def forward(self, input):
        """
        Args:
            input (batch_size, 2*in_channels, T)
        Returns:
            output (batch_size, 2*out_channels, T')
        """
        in_channels, out_channels = self.in_channels, self.out_channels
        kernel_size, stride = self.kernel_size, self.stride
        padding, dilation = self.padding, self.dilation
        groups = self.groups
        
        input_real, input_imag = input[:,:in_channels], input[:,in_channels:]
        output_real = F.conv1d(input_real, self.weight_real, stride=stride, padding=padding, groups=groups) - F.conv1d(input_imag, self.weight_imag, stride=stride, padding=padding, groups=groups)
        output_imag = F.conv1d(input_real, self.weight_imag, stride=stride, padding=padding, groups=groups) + F.conv1d(input_imag, self.weight_real, stride=stride, padding=padding, groups=groups)
        output = torch.cat([output_real, output_imag], dim=1)

        return output

class MultiDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None):
        super().__init__()

        self.kernel_size = kernel_size

        sections = []
        conv1d = []

        if type(in_channels) is int:
            assert groups is not None, "Specify groups"
            assert in_channels % groups == 0, "`in_channels` must be divisible by `groups`"

            self.groups = groups
            _in_channels = in_channels // groups

            for idx in range(groups):
                sections.append(_in_channels)
                conv1d.append(nn.Conv1d(_in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=2**idx))
        elif type(in_channels) is list:
            self.groups = len(in_channels)
            for idx, _in_channels in enumerate(in_channels):
                sections.append(_in_channels)
                conv1d.append(nn.Conv1d(_in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=2**idx))
        else:
            raise ValueError("Not support `in_channels`={}".format(in_channels))
        
        self.sections = sections
        self.conv1d = nn.ModuleList(conv1d)
    
    def forward(self, input):
        kernel_size = self.kernel_size

        input = torch.split(input, self.sections, dim=1)
        output = 0

        for idx in range(len(self.sections)):
            dilation = 2**idx
            padding = (kernel_size - 1) * dilation
            padding_left = padding // 2
            padding_right = padding - padding_left

            x = F.pad(input[idx], (padding_left, padding_right))
            output = output + self.conv1d[idx](x)

        return output


class MultiDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups=None):
        super().__init__()

        self.kernel_size = _pair(kernel_size)

        sections = []
        conv2d = []

        if type(in_channels) is int:
            assert groups is not None, "Specify groups"
            assert in_channels % groups == 0, "`in_channels` must be divisible by `groups`"

            self.groups = groups
            _in_channels = in_channels // groups

            for idx in range(groups):
                sections.append(_in_channels)
                conv2d.append(nn.Conv2d(_in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=2**idx))
        elif type(in_channels) is list:
            self.groups = len(in_channels)
            for idx, _in_channels in enumerate(in_channels):
                sections.append(_in_channels)
                conv2d.append(nn.Conv2d(_in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=2**idx))
        else:
            raise ValueError("Not support `in_channels`={}".format(in_channels))
        
        self.sections = sections
        self.conv2d = nn.ModuleList(conv2d)
    
    def forward(self, input):
        kernel_size = self.kernel_size

        input = torch.split(input, self.sections, dim=1)
        output = 0

        for idx in range(len(self.sections)):
            dilation = 2**idx
            padding_height = (kernel_size[0] - 1) * dilation
            padding_width = (kernel_size[1] - 1) * dilation
            padding_up = padding_height // 2
            padding_bottom = padding_height - padding_up
            padding_left = padding_width // 2
            padding_right = padding_width - padding_left

            x = F.pad(input[idx], (padding_left, padding_right, padding_up, padding_bottom))
            output = output + self.conv2d[idx](x)

        return output

def _test_multidilated_conv1d():
    torch.manual_seed(111)

    batch_size = 2
    in_channels, out_channels = [1, 1, 1], 4
    T = 16

    kernel_size = 3

    input = torch.randn(batch_size, sum(in_channels), T)
    conv1d = MultiDilatedConv1d(sum(in_channels), out_channels, kernel_size=kernel_size, groups=len(in_channels))
    print(conv1d)
    output = conv1d(input)
    print(input.size(), output.size())

def _test_multidilated_conv2d():
    torch.manual_seed(111)

    batch_size = 2
    in_channels, out_channels = [1, 1, 1], 4
    H, W = 16, 32

    kernel_size = 3

    input = torch.randn(batch_size, sum(in_channels), H, W)
    conv2d = MultiDilatedConv2d(sum(in_channels), out_channels, kernel_size=kernel_size, groups=len(in_channels))
    print(conv2d)
    output = conv2d(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    """
    batch_size = 4
    
    in_channels, hidden_channels = 32, 8
    
    T = 128
    kernel_size1d, stride1d = 3, 2
    input1d = torch.randint(0, 5, (batch_size, in_channels, T)).float()
    print(input1d.size())
    
    H, W = 32, 64
    kernel_size2d, stride2d = (3,3), (2,2)
    input2d = torch.randint(0, 10, (batch_size, in_channels, H, W), dtype=torch.float)
    print(input2d.size())
    """

    _test_multidilated_conv1d()
    print()

    _test_multidilated_conv2d()
    
