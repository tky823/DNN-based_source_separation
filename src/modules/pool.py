import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

"""
    Global max pooling
"""
def global_max_pool1d(input, keepdim=False):
    x = F.adaptive_max_pool1d(input, 1)

    if keepdim:
        output = x
    else:
        output = torch.flatten(x, start_dim=1)

    return output

def global_max_pool2d(input, keepdim=False):
    x = F.adaptive_max_pool2d(input, (1, 1))

    if keepdim:
        output = x
    else:
        output = torch.flatten(x, start_dim=1)

    return output

class GlobalMaxPool1d(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, input):
        """
        Args:
            input (batch_size, C, T)
        Returns:
            output (batch_size, C, 1) or (batch_size, C)
        """
        output = global_max_pool1d(input, keepdim=self.keepdim)

        return output

class GlobalMaxPool2d(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, 1, 1) or (batch_size, C)
        """
        output = global_max_pool2d(input, keepdim=self.keepdim)

        return output

"""
    Global average pooling
"""
def global_avg_pool1d(input, keepdim=False):
    x = F.adaptive_avg_pool1d(input, 1)

    if keepdim:
        output = x
    else:
        output = torch.flatten(x, start_dim=1)

    return output

def global_avg_pool2d(input, keepdim=False):
    x = F.adaptive_avg_pool2d(input, (1, 1))

    if keepdim:
        output = x
    else:
        output = torch.flatten(x, start_dim=1)

    return output

class GlobalAvgPool1d(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, input):
        """
        Args:
            input (batch_size, C, T)
        Returns:
            output (batch_size, C, 1) or (batch_size, C)
        """
        output = global_avg_pool1d(input, keepdim=self.keepdim)

        return output

class GlobalAvgPool2d(nn.Module):
    def __init__(self, keepdim=False):
        super().__init__()

        self.keepdim = keepdim

    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, 1, 1) or (batch_size, C)
        """
        output = global_avg_pool2d(input, keepdim=self.keepdim)

        return output

"""
    Stochastic pooling
"""
class StochasticPool1d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()

        if stride is None:
            stride = kernel_size

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        """
        Args:
            input (batch_size, C, T)
        Returns:
            output (batch_size, C, T'), where T' = (T-K)//S+1
        """
        K, S = self.kernel_size, self.stride

        batch_size, C, T = input.size()

        if (input < 0).any():
            raise ValueError("Place non-negative output function before Stochastic pooling.")
        elif (input==0).all():
            zero_flg = 1
        else:
            zero_flg = 0

        input = input.unsqueeze(dim=3) # -> (batch_size, C, T, 1)
        input = F.unfold(input, kernel_size=(K,1), stride=(S,1)) # -> (batch_size, C*K, T'), where T' = (T-K)//S+1
        input = input.view(batch_size*C, K, -1) # -> (batch_size*C, K, T')
        input = input.permute(0,2,1).contiguous() # -> (batch_size*C, T', K)
        input = input.view(-1, K) # -> (batch_size*C*T', K)

        weights = input # (batch_size*C*T', K)

        if zero_flg:
            weights = weights + 1
            # So weights' elements are all 1.

        if self.training:
            indices = torch.multinomial(weights, 1).view(-1) # -> (batch_size*C*T', 1)
            mask = torch.eye(K)[indices].float() # -> (batch_size*C*T', K)
            output = mask * input
        else:
            weights /= weights.sum(dim=1, keepdim=True) # -> (batch_size*C*T', K)
            output = weights * input # -> (batch_size*C*T', K)

        output = output.sum(dim=1) # -> (batch_size*C*T',)
        output = output.view(batch_size, C, -1) # -> (batch_size, C, T')

        return output

class StochasticPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size

        stride = _pair(stride)

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input):
        """
        Args:
            input (batch_size, C, H, W)
        Returns:
            output (batch_size, C, H', W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        """
        Kh, Kw = self.kernel_size
        Sh, Sw = self.stride

        batch_size, C, H, W = input.size()

        if (input < 0).any():
            raise ValueError("Place non-negative output function before Stochastic pooling.")
        elif (input==0).all():
            zero_flg = 1
        else:
            zero_flg = 0

        input = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride) # -> (batch_size, C*Kh*Kw, H'*W'), where H' = (H-Kh)//Sh+1, W' = (W-Kw)//Sw+1
        input = input.view(batch_size*C, Kh*Kw, -1) # -> (batch_size*C, Kh*Kw, H'*W')
        input = input.permute(0,2,1).contiguous() # -> (batch_size*C, H'*W', Kh*Kw)
        input = input.view(-1, Kh*Kw) # -> (batch_size*C*H'*W', Kh*Kw)

        weights = input # (batch_size*C*H'*W', Kh*Kw)

        if zero_flg:
            weights = weights + 1
            # So weights' elements are all 1.

        if self.training:
            indices = torch.multinomial(weights, 1).view(-1) # -> (batch_size*C*H'*W', 1)
            mask = torch.eye(Kh*Kw)[indices].float() # -> (batch_size*C*H'*W', Kh*Kw)
            output = mask * input
        else:
            weights /= weights.sum(dim=1, keepdim=True) # -> (batch_size*C*H'*W', Kh*Kw)
            output = weights * input # -> (batch_size*C*H'*W', Kh*Kw)

        output = output.sum(dim=1) # -> (batch_size*C*H'*W',)
        output = output.view(batch_size, C, -1) # -> (batch_size, C, H'*W')
        output = F.fold(output, kernel_size=(1,1), stride=self.stride, output_size=((H-Kh)//Sh+1,(W-Kw)//Sw+1))

        return output

def median_pool2d(input, kernel_size, stride=1, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    (kH, kW), (sH, sW) = _pair(kernel_size), _pair(stride)
    (pH, pW), (dH, dW) = _pair(padding), _pair(dilation)

    B, C, H_in, W_in = input.size()
    H_out = math.floor(((H_in + 2 * pH - dH * (kH - 1) - 1) / sH) + 1)
    W_out = math.floor(((W_in + 2 * pW - dW * (kW - 1) - 1) / sW) + 1)

    input = F.unfold(input, kernel_size, dilation, padding, stride)
    input = input.view(B, C, kH*kW, -1)
    output, _ = torch.median(input, dim=2, keepdim=True)
    output = output.view(B, C, H_out, W_out)

    if return_indices:
        raise NotImplementedError("Set return_indices=False.")

    if ceil_mode:
        raise NotImplementedError("Set ceil_mode=False.")

    return output

class MedianPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__()

        self.kernel_size, self.stride = kernel_size, stride
        self.padding, self.dilation = padding, dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        output = median_pool2d(input, self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, return_indices=self.return_indices, ceil_mode=self.ceil_mode)
        return output

"""
    Generalized mean pooling
    Reference:
        "Fine-tuning CNN Image Retrieval with No Human Annotation"
        See https://arxiv.org/abs/1711.02512
"""
def gem_pool2d(input, p=3, eps=1e-6):
    """
    Args:
        input <torch.Tensor>: (batch_size, num_features, height, width)
        p <torch.Tensor> or <float> or <int>: Coefficient of Lp norm.
        eps <float>: Flooring threshold.
    Returns:
        output <torch.Tensor>: (batch_size, num_features)
    """
    x = torch.clamp(input, min=eps)
    x = F.adaptive_avg_pool2d(x**p, 1)
    x = torch.flatten(x, start_dim=1)
    output = x ** (1 / p)

    return output

class GeMPool2d(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()

        self.p = nn.Parameter(torch.tensor(float(p)), requires_grad=True)
        self.eps = eps

    def forward(self, input):
        """
        Args:
            input: (batch_size, num_features, height, width)
        Returns:
            output: (batch_size, num_features)
        """
        output = gem_pool2d(input, p=self.p, eps=self.eps)

        return output

def _test_global_max_pool():
    batch_size, num_features, length, height, width = 1, 2, 5, 3, 4

    input1d = torch.randint(0, 10, (batch_size, num_features, length), dtype=torch.float)
    print(input1d)
    print(input1d.size())

    input2d = torch.randint(0, 10, (batch_size, num_features, height, width), dtype=torch.float)
    print(input2d)
    print(input2d.size())

    print("-"*10, "GlobalMaxPool1d", "-"*10)
    global_max_pool1d = GlobalMaxPool1d()
    output = global_max_pool1d(input1d)
    print(output)
    print(output.size())
    print()

    print("-"*10, "GlobalMaxPool2d", "-"*10)
    global_max_pool2d = GlobalMaxPool2d()
    output = global_max_pool2d(input2d)
    print(output)
    print(output.size())
    print()

def _test_global_avg_pool():
    batch_size, num_features, length, height, width = 1, 2, 5, 3, 4

    input1d = torch.randint(0, 10, (batch_size, num_features, length), dtype=torch.float)
    print(input1d)
    print(input1d.size())

    input2d = torch.randint(0, 10, (batch_size, num_features, height, width), dtype=torch.float)
    print(input2d)
    print(input2d.size())

    print("-"*10, "GlobalAvgPool1d", "-"*10)
    global_avg_pool1d = GlobalAvgPool1d()
    output = global_avg_pool1d(input1d)
    print(output)
    print(output.size())
    print()

    print("-"*10, "GlobalAvgPool2d", "-"*10)
    global_avg_pool2d = GlobalAvgPool2d()
    output = global_avg_pool2d(input2d)
    print(output)
    print(output.size())
    print()

def _test_stochastic_pool():
    batch_size, num_features, length, height, width = 1, 2, 5, 3, 4
    kernel_size1d, kernel_size2d = 3, (2, 3)
    stride1d, stride2d = 2, (1, 1)

    input1d = torch.randint(0, 10, (batch_size, num_features, length), dtype=torch.float)
    print(input1d)
    print(input1d.size())

    input2d = torch.randint(0, 10, (batch_size, num_features, height, width), dtype=torch.float)
    print(input2d)
    print(input2d.size())

    print("-"*10, "StochasticPool1d", "-"*10)
    stochastic_pool1d = StochasticPool1d(kernel_size=kernel_size1d, stride=stride1d)
    output = stochastic_pool1d(input1d)
    print(output)
    print(output.size())

    stochastic_pool1d.eval()
    output = stochastic_pool1d(input1d)
    print(output)
    print(output.size())
    print()

    print("-"*10, "StochasticPool2d", "-"*10)
    stochastic_pool2d = StochasticPool2d(kernel_size=kernel_size2d, stride=stride2d)
    output = stochastic_pool2d(input2d)
    print(output)
    print(output.size())

    stochastic_pool2d.eval()
    output = stochastic_pool2d(input2d)
    print(output)
    print(output.size())
    print()

def _test_medial_pool2d():
    B, C, H, W = 4, 3, 6, 8
    input = torch.randint(0, 10, (B, C, H, W), dtype=torch.float)
    output = median_pool2d(input, kernel_size=3, stride=2)

    print(input.size(), output.size())

    for _input, _output in zip(input[0], output[0]):
        print(_input)
        print(_output)
        print()

if __name__ == "__main__":
    torch.manual_seed(111)

    # Global max pooling
    print("="*10, "Global max pooling", "="*10)
    _test_global_max_pool()

    # Global average pooling
    print("="*10, "Global average pooling", "="*10)
    _test_global_avg_pool()

    # Stochastic pooling
    print("="*10, "Stochastic pooling", "="*10)
    _test_stochastic_pool()

    # Median pooling
    print("="*10, "Median pooling", "="*10)
    print("-"*10, "MedianPool2d", "-"*10)
    _test_medial_pool2d()