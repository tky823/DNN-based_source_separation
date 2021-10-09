import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.tdf import TDF2d, MultiheadTDF2d

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
    def __init__(self, in_channels, embed_dim, hidden_dim, n_bins, bottleneck_bins=None, num_heads=2, bias=False, nonlinear='relu'):
        super().__init__()

        transform_query = nn.Linear(embed_dim, hidden_dim)
        transform_value = nn.Sequential(
            TDF2d(in_channels, n_bins, bottleneck_bins, bias=bias, nonlinear=nonlinear),
            MultiheadTDF2d(in_channels, bottleneck_bins, n_bins, num_heads=num_heads, bias=bias, nonlinear=nonlinear, stack_dim=2)
        )

        self.lasaft = LaSAFT(hidden_dim, transform_query, transform_value, num_heads=num_heads)

    def forward(self, input, embedding):
        output = self.lasaft(input, embedding=embedding)

        return output

def _test_tfc_lasaft():
    batch_size = 3
    in_channels = 2
    embed_dim, hidden_dim = 8, 16
    n_bins, bottleneck_bins = 129, 32
    num_heads = 4
    n_frames = 12

    input, embedding = torch.randn((batch_size, in_channels, n_bins, n_frames), dtype=torch.float), torch.randn((batch_size, embed_dim), dtype=torch.float)

    model = TFCLaSAFT(in_channels, embed_dim, hidden_dim, n_bins, bottleneck_bins=bottleneck_bins, num_heads=num_heads)
    
    output = model(input, embedding=embedding)

    print(model)
    print(input.size(), output.size())

if __name__ == '__main__':
    _test_tfc_lasaft()

