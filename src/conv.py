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
        
        self.pointwise_conv2d = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=bias)
        self.depthwise_conv2d = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, groups=out_channels, bias=bias)
        

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

if __name__ == '__main__':
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
    
