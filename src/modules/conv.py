import math

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
        output = self.depthwise_conv1d(x)

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
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=None):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilations = []
        self.bias = bias

        self.sections = []
        weights = []
        biases = []

        if type(in_channels) is int:
            assert groups is not None, "Specify groups"
            assert in_channels % groups == 0, "`in_channels` must be divisible by `groups`"

            self.groups = groups
            _in_channels = in_channels // groups

            for idx in range(groups):
                dilation = 2**idx
                self.dilations.append(dilation)
                self.sections.append(_in_channels)
                weights.append(torch.Tensor(out_channels, _in_channels, kernel_size))
                if self.bias:
                    biases.append(torch.Tensor(out_channels,))
        elif type(in_channels) is list:
            self.groups = len(in_channels)
            for idx, _in_channels in enumerate(in_channels):
                dilation = 2**idx
                self.dilations.append(dilation)
                self.sections.append(_in_channels)
                weights.append(torch.Tensor(out_channels, _in_channels, kernel_size))
                if self.bias:
                    biases.append(torch.Tensor(out_channels,))
        else:
            raise ValueError("Not support `in_channels`={}".format(in_channels))

        weights = torch.cat(weights, dim=1)
        self.weights = nn.Parameter(weights)

        if self.bias:
            biases = torch.cat(biases, dim=0)
            self.biases = nn.Parameter(biases)

        self._reset_parameters()

    def forward(self, input):
        kernel_size = self.kernel_size

        weights = torch.split(self.weights, self.sections, dim=1)
        if self.bias:
            biases = torch.split(self.biases, self.out_channels, dim=0)
        else:
            biases = [None] * len(weights)

        input = torch.split(input, self.sections, dim=1)
        output = 0

        for idx in range(len(self.sections)):
            dilation = 2**idx
            padding = (kernel_size - 1) * dilation
            padding_left = padding // 2
            padding_right = padding - padding_left

            x = F.pad(input[idx], (padding_left, padding_right))
            output = output + F.conv1d(x, weight=weights[idx], bias=biases[idx], stride=1, dilation=dilation)

        return output

    def _reset_parameters(self):
        nn.modules.conv.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.modules.conv.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.modules.conv.init.uniform_(self.biases, -bound, bound)

    def extra_repr(self):
        s = "{}, {}".format(sum(self.sections), self.out_channels)
        s += ", kernel_size={kernel_size}, dilations={dilations}".format(kernel_size=self.kernel_size, dilations=self.dilations)
        if not self.bias:
            s += ", bias=False"
        return s

class MultiDilatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups=None):
        super().__init__()

        kernel_size = _pair(kernel_size)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilations = []
        self.bias = bias

        self.sections = []
        weights = []
        biases = []

        if type(in_channels) is int:
            assert groups is not None, "Specify groups"
            assert in_channels % groups == 0, "`in_channels` must be divisible by `groups`"

            self.groups = groups
            _in_channels = in_channels // groups

            for idx in range(groups):
                dilation = _pair(2**idx)
                self.dilations.append(dilation)
                self.sections.append(_in_channels)
                weights.append(torch.Tensor(out_channels, _in_channels, *kernel_size))
                if self.bias:
                    biases.append(torch.Tensor(out_channels,))
        elif type(in_channels) is list:
            self.groups = len(in_channels)
            for idx, _in_channels in enumerate(in_channels):
                dilation = _pair(2**idx)
                self.dilations.append(dilation)
                self.sections.append(_in_channels)
                weights.append(torch.Tensor(out_channels, _in_channels, *kernel_size))
                if self.bias:
                    biases.append(torch.Tensor(out_channels,))
        else:
            raise ValueError("Not support `in_channels`={}".format(in_channels))

        weights = torch.cat(weights, dim=1)
        self.weights = nn.Parameter(weights)

        if self.bias:
            biases = torch.cat(biases, dim=0)
            self.biases = nn.Parameter(biases)

        self._reset_parameters()

    def forward(self, input):
        kernel_size = self.kernel_size

        weights = torch.split(self.weights, self.sections, dim=1)
        if self.bias:
            biases = torch.split(self.biases, self.out_channels, dim=0)
        else:
            biases = [None] * len(weights)

        input = torch.split(input, self.sections, dim=1)
        output = 0

        for idx in range(len(self.sections)):
            dilation = self.dilations[idx]
            padding_height = (kernel_size[0] - 1) * dilation[0]
            padding_width = (kernel_size[1] - 1) * dilation[1]
            padding_up = padding_height // 2
            padding_bottom = padding_height - padding_up
            padding_left = padding_width // 2
            padding_right = padding_width - padding_left

            x = F.pad(input[idx], (padding_left, padding_right, padding_up, padding_bottom))
            output = output + F.conv2d(x, weight=weights[idx], bias=biases[idx], stride=(1,1), dilation=dilation)

        return output

    def _reset_parameters(self):
        nn.modules.conv.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.modules.conv.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.modules.conv.init.uniform_(self.biases, -bound, bound)

    def extra_repr(self):
        s = "{}, {}".format(sum(self.sections), self.out_channels)
        s += ", kernel_size={kernel_size}, dilations={dilations}".format(kernel_size=self.kernel_size, dilations=self.dilations)
        if not self.bias:
            s += ", bias=False"
        return s

"""
    Quantization
"""
class QuantizableConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        assert in_channels == out_channels and padding == 0 and dilation == 1 and groups == 1 and padding_mode == 'zeros', "Only supports the operation convertible into torch.nn.Conv1d."

        num_features = in_channels
        self.backend = nn.Conv1d(num_features, num_features, kernel_size, stride=1, padding=padding, dilation=1, groups=groups, bias=bias, padding_mode=padding_mode)

        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, input):
        kernel_size, stride = self.kernel_size, self.stride

        input_pad = F.pad(input.unsqueeze(dim=-1), (0, stride - 1))
        input_pad = input_pad.view(*input_pad.size()[:-2], -1)
        input_pad = F.pad(input_pad, (kernel_size - 1, kernel_size - stride))

        output = self.backend(input_pad)

        return output

class QuantizableConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        kernel_size, stride = _pair(kernel_size), _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        assert in_channels == out_channels and padding == (0, 0) and dilation == (1, 1) and groups == 1 and padding_mode == 'zeros', "Only supports the operation convertible into torch.nn.Conv2d."

        num_features = in_channels
        self.backend = nn.Conv2d(num_features, num_features, kernel_size, stride=1, padding=padding, dilation=1, groups=groups, bias=bias, padding_mode=padding_mode)

        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, input):
        (Kh, Kw), (Sh, Sw) = self.kernel_size, self.stride

        input_pad = F.pad(input.unsqueeze(dim=-1), (0, Sw - 1, 0, Sh - 1))
        input_pad = input_pad.view(*input_pad.size()[:-2], -1)
        input_pad = F.pad(input_pad, (Kw - 1, Kw - Sw, Kh - 1, Kh - Sh))

        output = self.backend(input_pad)

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
    print()

    conv1d = MultiDilatedConv1d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
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
    print()

    conv2d = MultiDilatedConv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
    print(conv2d)
    output = conv2d(input)
    print(input.size(), output.size())
    print()

if __name__ == '__main__':
    _test_multidilated_conv1d()
    print()

    _test_multidilated_conv2d()
