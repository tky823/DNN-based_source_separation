import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cunet import TDF2d, MultiheadTDF2d
from models.cunet import TFC2d

EPS = 1e-12

"""
Latent Source Attentive Frequency Transformation
Reference: "LaSAFT: Latent Source Attentive Frequency Transformation For Conditioned Source Separation"
"""

class LaSAFT(nn.Module):
    """
    Latent Source Attentive Frequency Transformation
    """
    def __init__(self, hidden_dim, transform_query, transform_value, num_heads=2):
        """
        Args:
            transform_query <nn.Module>
            transform_value <nn.Module>
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        self.key = nn.Parameter(torch.Tensor(hidden_dim, num_heads), requires_grad=True)
        self.transform_query = transform_query
        self.transform_value = transform_value

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.key)

    def forward(self, input, embedding):
        """
        Args:
            input (batch_size, in_channels, n_bins, n_frames)
            embedding (batch_size, embed_dim)
        Returns:
            output (batch_size, in_channels, n_bins, n_frames)
        """
        dk_sqrt = math.sqrt(self.hidden_dim)

        key = self.key # (hidden_dim, num_heads)
        query = self.transform_query(embedding) # (batch_size, hidden_dim)
        value = self.transform_value(input) # (batch_size, in_channels, num_heads, n_bins, n_frames)

        qk = torch.matmul(query, key) / dk_sqrt
        atten = F.softmax(qk, dim=-1) # (batch_size, num_heads)
        atten = atten.unsqueeze(dim=1).unsqueeze(dim=3).unsqueeze(dim=4) # (batch_size, 1, num_heads, 1, 1)

        output = atten * value # (batch_size, in_channels, num_heads, n_bins, n_frames)
        output = output.sum(dim=2) # (batch_size, in_channels, n_bins, n_frames)

        return output

class TFCLaSAFT(nn.Module):
    def __init__(self, in_channels, growth_rate, embed_dim, hidden_dim, n_bins, bottleneck_bins=None, kernel_size=None, num_layers=2, num_heads=2, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        self.tfc2d = TFC2d(in_channels, growth_rate=growth_rate, kernel_size=kernel_size, num_layers=num_layers, nonlinear=nonlinear)

        # LaSAFT
        transform_query = nn.Linear(embed_dim, hidden_dim)
        transform_value = nn.Sequential(
            TDF2d(growth_rate, n_bins, bottleneck_bins, nonlinear=nonlinear, bias=bias, eps=eps),
            MultiheadTDF2d(growth_rate, bottleneck_bins, n_bins, num_heads=num_heads, nonlinear=nonlinear, bias=bias, stack_dim=2, eps=eps)
        )
        self.lasaft = LaSAFT(hidden_dim, transform_query, transform_value, num_heads=num_heads)

    def forward(self, input, embedding):
        x = self.tfc2d(input)
        output = x + self.lasaft(x, embedding=embedding)

        return output

class TFCLightSAFT(nn.Module):
    def __init__(self, in_channels, growth_rate, embed_dim, hidden_dim, n_bins, bottleneck_bins=None, kernel_size=None, num_layers=2, num_heads=2, nonlinear='relu', bias=False, eps=EPS):
        super().__init__()

        self.tfc2d = TFC2d(in_channels, growth_rate=growth_rate, kernel_size=kernel_size, num_layers=num_layers, nonlinear=nonlinear)

        # LaSAFT
        transform_query = nn.Linear(embed_dim, hidden_dim)
        transform_value = MultiheadTDF2d(growth_rate, in_bins=n_bins, out_bins=bottleneck_bins, num_heads=num_heads, nonlinear=nonlinear, bias=bias, stack_dim=2, eps=eps)
        self.lasaft = LaSAFT(hidden_dim, transform_query, transform_value, num_heads=num_heads)

        self.tdf2d = TDF2d(growth_rate, in_bins=bottleneck_bins, out_bins=n_bins, nonlinear=nonlinear, bias=bias, eps=eps)

    def forward(self, input, embedding):
        x = self.tfc2d(input)
        x_lasaft = self.lasaft(x, embedding=embedding)
        output = x + self.tdf2d(x_lasaft)

        return output

def _test_tfc_lasaft():
    batch_size = 3
    in_channels = 2
    n_bins, n_frames = 129, 12
    growth_rate = 5
    kernel_size, num_layers = (3, 5), 2

    embed_dim, hidden_dim = 8, 16
    bottleneck_bins = 32
    num_heads = 4

    input, embedding = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float), torch.randn((batch_size, embed_dim), dtype=torch.float)

    model = TFCLaSAFT(in_channels, growth_rate=growth_rate, embed_dim=embed_dim, hidden_dim=hidden_dim, n_bins=n_bins, bottleneck_bins=bottleneck_bins, kernel_size=kernel_size, num_layers=num_layers, num_heads=num_heads)
    output = model(input, embedding=embedding)

    print(model)
    print(input.size(), output.size())

def _test_tfc_light_saft():
    batch_size = 3
    in_channels = 2
    n_bins, n_frames = 129, 12
    growth_rate = 5
    kernel_size, num_layers = (3, 5), 2

    embed_dim, hidden_dim = 8, 16
    bottleneck_bins = 32
    num_heads = 4

    input, embedding = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float), torch.randn((batch_size, embed_dim), dtype=torch.float)

    model = TFCLightSAFT(in_channels, growth_rate=growth_rate, embed_dim=embed_dim, hidden_dim=hidden_dim, n_bins=n_bins, bottleneck_bins=bottleneck_bins, kernel_size=kernel_size, num_layers=num_layers, num_heads=num_heads)
    output = model(input, embedding=embedding)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "LaSAFT", "="*10)
    _test_tfc_lasaft()
    print()

    print("="*10, "LightSAFT", "="*10)
    _test_tfc_light_saft()
