import itertools

import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class MultiDissimilarityLoss(nn.Module):
    def  __init__(self, n_sources=None, eps=EPS):
        super().__init__()

        left, right = [],[]

        for pair in itertools.combinations(range(n_sources), 2):
            _left, _right = pair
            left.append(_left)
            right.append(_right)
        
        self.left, self.right = right
        self.n_combinations = len(left)
        self.eps = eps

    def forward(self, input, batch_mean=True):
        """
        Args:
            input (batch_size, n_sources, *)
        """
        n_dims = input.dim()
        dims = list(range(n_dims))

        input_permuted = input.permute(1, 0, *dims[2:])

        left, right = self.left, self.right

        input_left, input_right = input_permuted[left], input_permuted[right]
        loss = - F.cosine_similarity(input_left, input_right, dim=0, eps=self.eps)

        dims = tuple(range(1, n_dims-1))
        loss = loss.sum(dim=dims) / self.n_combinations

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
