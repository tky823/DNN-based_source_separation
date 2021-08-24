import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class SimilarityLoss(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps
    
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input: (batch_size, n_sources, n_channels, n_frames)
            output: (batch_size, n_sources, n_channels, n_frames)
        """
        loss = F.cosine_similarity(input, target, dim=2, eps=self.eps)
        loss = loss.sum(dim=2)
        loss = loss.mean(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class NegSimilarityLoss(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()

        self.eps = eps
    
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input: (batch_size, n_sources, n_channels, n_frames)
            output: (batch_size, n_sources, n_channels, n_frames)
        """
        loss = - F.cosine_similarity(input, target, dim=2, eps=self.eps)
        loss = loss.sum(dim=2)
        loss = loss.mean(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss


class MultiDissimilarityLoss(nn.Module):
    def  __init__(self, n_sources=None, eps=EPS):
        super().__init__()

        left, right = [],[]

        for pair in itertools.combinations(range(n_sources), 2):
            _left, _right = pair
            left.append(_left)
            right.append(_right)
        
        self.left, self.right = left, right
        self.n_combinations = len(left)
        self.eps = eps

    def forward(self, input, batch_mean=True):
        """
        Args:
            input (batch_size, n_sources, n_channels, n_frames)
        """
        input_permuted = input.permute(1, 0, 2, 3).contiguous()

        left, right = self.left, self.right

        input_left, input_right = torch.abs(input_permuted[left]), torch.abs(input_permuted[right])
        loss = - F.cosine_similarity(input_left, input_right, dim=2, eps=self.eps)

        loss = loss.permute(1, 0, 2).contiguous() / self.n_combinations
        loss = loss.sum(dim=2)
        loss = loss.mean(dim=1)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class MultiLoss(nn.Module):
    def  __init__(self, metrics, weights):
        super().__init__()

        self.metrics = nn.ModuleDict(metrics)
        self.weights = weights
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
