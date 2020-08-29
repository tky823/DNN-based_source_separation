import torch
import torch.nn as nn
import torch.nn.functional as F

EPS=1e-12

"""
    Global layer normalization
    See "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
    https://arxiv.org/abs/1809.07454
"""

class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()
        
        self.norm = nn.GroupNorm(1, num_features, eps=eps)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, C, *)
        Returns:
            output (batch_size, C, *)
        """
        output = self.norm(input)
        
        return output

"""
    Cumulative layer normalization
    See "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
    https://arxiv.org/abs/1809.07454
"""

class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()
        
        self.eps = eps

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()
        
    def forward(self, input):
        eps = self.eps
        batch_size, C, T = input.size()
        
        step_sum = input.sum(dim=1) # -> (batch_size, T)
        input_pow = input**2
        step_pow_sum = input_pow.sum(dim=1) # -> (batch_size, T)
        cum_sum = torch.cumsum(step_sum, dim=1) # -> (batch_size, T)
        cum_squared_sum = torch.cumsum(step_pow_sum, dim=1) # -> (batch_size, T)
        
        cum_num = torch.arange(C, C*(T+1), C, dtype=torch.float) # -> (T, ): [C, 2*C, ..., T*C]
        cum_mean = cum_sum / cum_num # (batch_size, T)
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean**2
        
        cum_mean = cum_mean.unsqueeze(dim=1)
        cum_var = cum_var.unsqueeze(dim=1)
        
        output = (input - cum_mean) / (torch.sqrt(cum_var) + eps) * self.gamma + self.beta
        
        return output
