import torch
import torch.nn as nn

"""
"Many-Speakers Single Channel Speech Separation with Optimal Permutation Training"
"""

class HungarianLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, batch_mean=True):
        raise NotImplementedError