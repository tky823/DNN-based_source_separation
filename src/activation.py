import torch
import torch.nn as nn
import torch.nn.functional as F

EPS=1e-12

class ModReLU1d(nn.Module):
    def __init__(self, n_units, eps=EPS):
        super().__init__()
        
        self.n_units = n_units
        self.eps = eps
        
        self.bias = nn.Parameter(torch.Tensor((1,n_units,1)))
        
        self._reset_parameters()
        
    
    def forward(self, input):
        n_units = self.n_units
        
        real, imag = input[:,:n_units//2], input[:,n_units//2:]
        magnitude = torch.sqrt(real**2 + imag**2)
        output_magnitude = magnitude + self.bias
        ratio = output_magnitude / (magnitude + self.eps)
        ratio = torch.where(output_magnitude >= 0, ratio, torch.zeros_like(magnitude))
        real, imag = ratio * real, ratio * imag
        output = torch.cat([real, imag], dim=1)
        
        return output
    
    def _reset_parameters(self):
        self.bias.data.zero_()
        

"""
    See "Deep Complex Networks"
    https://arxiv.org/abs/1705.09792
"""

class ComplexReLU(nn.Module):
    """
    Same as ReLU
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        """
        Args:
            input (*) <torch.Tensor>:
        Returns:
            output (*) <torch.Tensor>:
        """
        output = F.ReLU(input)
        
        return output
"""
    z ReLU
    See "On complex valued convolutional neural networks" or "Deep Complex Networks"
    https://arxiv.org/abs/1602.09046
    https://arxiv.org/abs/1705.09792
"""
class ZReLU1d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        n_units = input.size(1)
        real, imag = input[:,:n_units//2], input[:,n_units//2:]
        
        condition = torch.logical_and(real > 0, imag > 0)
        output_real = torch.where(condition, real, torch.zeros_like(real))
        output_imag = torch.where(condition, imag, torch.zeros_like(imag))
        output = torch.cat([output_real, output_imag], dim=1)
        
        return output
