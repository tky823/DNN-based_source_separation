import torch
import torch.nn as nn

"""
Reference: D3Net: Densely connected multidilated DenseNet for music source separation
See https://arxiv.org/abs/2010.01733
"""

EPS=1e-12

class D3Net(nn.Module):
    def __init__(self, eps=EPS, **kwargs):
        super().__init__()

        self.eps = eps

        raise NotImplementedError("Implement D3Net")
    
    def forward(self, input):
        raise NotImplementedError("Implement D3Net")