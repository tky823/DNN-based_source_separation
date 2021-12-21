import itertools

import torch
import torch.nn as nn

EPS = 1e-12

class CombinationLoss(nn.Module):
    """
    Combination Loss for Multi Sources
    """
    def __init__(self, criterion, combination_dim=1, min_pair=1, max_pair=None):
        super().__init__()

        self.criterion = criterion

        self.combination_dim = combination_dim
        self.min_pair, self.max_pair = min_pair, max_pair

    def forward(self, input, target, reduction='mean', batch_mean=True):
        assert target.size() == input.size(), "input.size() are expected same."

        combination_dim = self.combination_dim
        min_pair, max_pair = self.min_pair, self.max_pair

        n_sources = input.size(combination_dim)

        if max_pair is None:
            max_pair = n_sources - 1

        input = torch.unbind(input, dim=combination_dim)
        target = torch.unbind(target, dim=combination_dim)

        loss = []

        for _n_sources in range(min_pair, max_pair + 1):
            for pair_indices in itertools.combinations(range(n_sources), _n_sources):
                _input, _target = [], []
                for idx in pair_indices:
                    _input.append(input[idx])
                    _target.append(target[idx])
                _input, _target = torch.stack(_input, dim=0), torch.stack(_target, dim=0)
                _input, _target = _input.sum(dim=0), _target.sum(dim=0)

                loss_pair = self.criterion(_input, _target, batch_mean=batch_mean)
                loss.append(loss_pair)

        dim = combination_dim - 1 if batch_mean else combination_dim
        loss = torch.stack(loss, dim=dim)

        if reduction == 'mean':
            loss = loss.mean(dim=dim)
        elif reduction == 'sum':
            loss = loss.sum(dim=dim)

        return loss

def _test_cl():
    batch_size = 3
    n_sources = 4
    in_channels, T = 2, 32

    input = torch.randn(batch_size, n_sources, in_channels, T)
    target = torch.randn(batch_size, n_sources, in_channels, T)

    criterion = NegWeightedSDR()
    combination_criterion = CombinationLoss(criterion, min_pair=1, max_pair=n_sources-1)

    loss = combination_criterion(input, target, batch_mean=False)
    print(loss)

if __name__ == '__main__':
    from criterion.sdr import NegWeightedSDR

    torch.manual_seed(111)

    print("="*10, "Combination Loss", "="*10)
    _test_cl()