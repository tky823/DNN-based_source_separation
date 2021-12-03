import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

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

    _test_medial_pool2d()