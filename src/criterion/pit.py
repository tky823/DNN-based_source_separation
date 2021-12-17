import itertools

import torch
import torch.nn as nn

"""
    Permutation invariant training
"""
def pit(criterion, input, target, n_sources=None, patterns=None, batch_mean=True):
    """
    Args:
        criterion <callable>
        input (batch_size, n_sources, *)
        output (batch_size, n_sources, *)
    Returns:
        loss (batch_size,): minimum loss for each data
        pattern (batch_size,): permutation indices
    """
    if patterns is None:
        if n_sources is None:
            n_sources = input.size(1)
        patterns = list(itertools.permutations(range(n_sources)))
        patterns = torch.Tensor(patterns).long()

    P = len(patterns)
    possible_loss = []

    for idx in range(P):
        pattern = patterns[idx]
        loss = criterion(input, target[:, pattern], batch_mean=False)
        possible_loss.append(loss)

    possible_loss = torch.stack(possible_loss, dim=1)

    # possible_loss (batch_size, P)
    if hasattr(criterion, "maximize") and criterion.maximize:
        loss, indices = torch.max(possible_loss, dim=1) # loss (batch_size,), indices (batch_size,)
    else:
        loss, indices = torch.min(possible_loss, dim=1) # loss (batch_size,), indices (batch_size,)

    if batch_mean:
        loss = loss.mean(dim=0)

    return loss, patterns[indices]

class PIT(nn.Module):
    def __init__(self, criterion, n_sources):
        """
        Args:
            criterion <callable>: criterion is expected acceptable (input, target, batch_mean) when called.
        """
        super().__init__()

        self.criterion = criterion
        patterns = list(itertools.permutations(range(n_sources)))
        self.patterns = torch.Tensor(patterns).long()

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, n_sources, *)
            target (batch_size, n_sources, *)
        Returns:
            loss (batch_size,): minimum loss for each data
            pattern (batch_size,): permutation indices
        """
        loss, pattern = pit(self.criterion, input, target, patterns=self.patterns, batch_mean=batch_mean)

        return loss, pattern

class PIT1d(PIT):
    def __init__(self, criterion, n_sources):
        """
        Args:
            criterion <callable>: criterion is expected acceptable (input, target, batch_mean) when called.
        """
        super().__init__(criterion, n_sources)

class PIT2d(PIT):
    def __init__(self, criterion, n_sources):
        """
        Args:
            criterion <callable>: criterion is expected acceptable (input, target, batch_mean) when called.
        """
        super().__init__(criterion, n_sources)

class ORPIT(nn.Module):
    """
    One-and-Rest permutation invariant training
    """
    def __init__(self, criterion):
        super().__init__()

        self.criterion = criterion
        patterns = list(itertools.permutations(range(2))) # 2 means 'one' and 'rest'
        self.patterns = torch.Tensor(patterns).long()

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, 2, *)
            target <packed_sequence>: `n_sources` is different for every input.
            n_sources (batch_size,): The number of sources per input. If `None`, the number is regarded same among the batch.
        Returns:
            loss (batch_size,): minimum loss for each data
            indices (batch_size,): most possible indices
        """
        assert input.size(1) == 2, "input.size() is expected (batch_size, 2, *), but given {}".format(input.size())

        criterion = self.criterion

        if type(target) is torch.Tensor:
            batch_size = target.size(0)
            lens_unpacked = [target.size(1)] * batch_size
        else:
            target, lens_unpacked = nn.utils.rnn.pad_packed_sequence(target, batch_first=True)

        # TODO: batch process

        batch_size = input.size(0)
        batch_loss, batch_indices = None, None

        for batch_idx in range(batch_size):
            n_sources = lens_unpacked[batch_idx] # <int>
            _input, _target = input[batch_idx: batch_idx + 1], target[batch_idx: batch_idx + 1, : n_sources] # (1, 2, *), (1, n_sources, *)
            input_one, input_rest = torch.unbind(_input, dim=1) # (1, *), (1, *)

            possible_loss = None

            for idx in range(n_sources):
                mask_one = torch.zeros_like(_target)
                mask_one[:, idx] = 1.0
                mask_rest = torch.ones_like(_target) - mask_one
                target_one = torch.sum(mask_one * _target, dim=1) # (1, *)
                target_rest = torch.sum(mask_rest * _target, dim=1) # (1, *)

                loss_one = criterion(input_one, target_one, batch_mean=False)
                loss_rest = criterion(input_rest, target_rest, batch_mean=False)
                loss = loss_one + loss_rest / (n_sources - 1)

                if possible_loss is None:
                    possible_loss = loss
                else:
                    possible_loss = torch.cat([possible_loss, loss], dim=0)

            if hasattr(criterion, "maximize") and criterion.maximize:
                loss, indices = torch.max(possible_loss, dim=0, keepdim=True) # loss (1,), indices (1,)
            else:
                loss, indices = torch.min(possible_loss, dim=0, keepdim=True) # loss (1,), indices (1,)

            if batch_loss is None:
                batch_loss = loss
                batch_indices = indices
            else:
                batch_loss = torch.cat([batch_loss, loss], dim=0)
                batch_indices = torch.cat([batch_indices, indices], dim=0)

        if batch_mean:
            batch_loss = batch_loss.mean(dim=0)

        return batch_loss, batch_indices

def sinkpit(criterion, input, target, n_sources=None, coldness=1e+0, iteration=10, batch_mean=True):    
    if n_sources is None:
        n_sources = input.size(1)

    batch_size = input.size(0)

    input_size, target_size = input.size()[2:], target.size()[2:]
    input, target = input.unsqueeze(dim=2).expand(-1, -1, n_sources, -1).contiguous(), target.unsqueeze(dim=1).expand(-1, n_sources, -1, -1).contiguous()
    input, target = input.view(batch_size * n_sources * n_sources, *input_size), target.view(batch_size * n_sources * n_sources, *target_size)
    possible_loss = criterion(input, target, batch_mean=False)
    possible_loss = possible_loss.view(batch_size, n_sources, n_sources)

    if hasattr(criterion, "maximize") and criterion.maximize:
        possible_loss = - possible_loss

    Z = - coldness * possible_loss

    for idx in range(iteration):
        Z = Z - torch.logsumexp(Z, dim=1, keepdim=True)
        Z = Z - torch.logsumexp(Z, dim=2, keepdim=True)

    permutaion_matrix = torch.exp(Z)
    loss = torch.sum((possible_loss + Z / coldness) * permutaion_matrix, dim=(1,2))

    if hasattr(criterion, "maximize") and criterion.maximize:
        loss = - loss

    if batch_mean:
        loss = loss.mean(dim=0)

    return loss, permutaion_matrix

class SinkPIT(nn.Module):
    """
    "Towards Listening to 10 People Simultaneously: An Efficient Permutation Invariant Training of Audio Source Separation Using Sinkhorn's Algorithm"
    See https://arxiv.org/abs/2010.11871
    """
    def __init__(self, criterion, n_sources=None, coldness=1, iteration=10):
        super().__init__()

        self.criterion = criterion
        self.n_sources = n_sources

        self.coldness = coldness
        self.iteration = iteration

    def forward(self, input, target, batch_mean=True):
        loss, permutation_matrix = sinkpit(self.criterion, input, target, n_sources=self.n_sources, coldness=self.coldness, iteration=self.iteration, batch_mean=batch_mean)
        pattern = torch.argmax(permutation_matrix, dim=2)

        return loss, pattern

class ProbPIT(nn.Module):
    """
    "Probabilistic permutation invariant training for speech separation"
    See https://arxiv.org/abs/1908.01768
    """
    def __init__(self):
        super().__init__()

    def forward(self, input, target, batch_mean=True):
        pass

def _test_pit():
    torch.manual_seed(111)

    batch_size, C, T = 4, 2, 1024
    input = torch.randint(2, (batch_size, C, T), dtype=torch.float)
    target = torch.randint(2, (batch_size, C, T), dtype=torch.float)

    print('-'*10, "SI-SDR", '-'*10)
    criterion = SISDR()
    pit_criterion = PIT1d(criterion, n_sources=C)
    loss, pattern = pit_criterion(input, target)

    print(loss)
    print(pattern)
    print()

    print('-'*10, "Squared error (customized loss)", '-'*10)
    def squared_error(input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *, T)
            target (batch_size, *, T)
        Returns:
            loss () or (batch_size)
        """
        loss = torch.sum((input - target)**2, dim=-1)
        n_dim = loss.dim()
        dim = tuple(range(1, n_dim))
        loss = loss.mean(dim=dim)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    pit_criterion = PIT1d(squared_error, n_sources=C)
    loss, pattern = pit_criterion(input, target)

    print(loss)
    print(pattern)

def _test_orpit():
    torch.manual_seed(111)

    T = 6

    print('-'*10, "SI-SDR", '-'*10)
    permutations = [[2, 1, 0], [0, 1], [0, 2, 1], [1, 0]]
    input, target = [], []

    for indice in permutations:
        n_sources = len(indice)
        _input = torch.randint(5, (n_sources, T), dtype=torch.float)
        _target = _input[torch.Tensor(indice).long()]

        _input_one, _input_rest = torch.split(_input, [1, n_sources - 1], dim=0)
        _input_rest = _input_rest.sum(dim=0, keepdim=True)
        _input = torch.cat([_input_one, _input_rest], dim=0)

        input.append(_input)
        target.append(_target)

    input = nn.utils.rnn.pad_sequence(input, batch_first=True)
    target = nn.utils.rnn.pack_sequence(target, enforce_sorted=False)

    criterion = SISDR()
    pit_criterion = ORPIT(criterion)
    loss, indices = pit_criterion(input, target)

    print(loss)
    print(indices)
    print()

    print('-'*10, "L1Loss", '-'*10)
    permutations = [[2, 3, 1, 0], [1, 0], [0, 1, 2], [2, 1, 0], [0, 1]]
    input, target = [], []

    for indice in permutations:
        n_sources = len(indice)
        _input = torch.randint(5, (n_sources, T), dtype=torch.float)
        _target = _input[torch.Tensor(indice).long()]

        _input_one, _input_rest = torch.split(_input, [1, n_sources - 1])
        _input_rest = _input_rest.sum(dim=0, keepdim=True)
        _input = torch.cat([_input_one, _input_rest], dim=0)

        input.append(_input)
        target.append(_target)

    input = nn.utils.rnn.pad_sequence(input, batch_first=True)
    target = nn.utils.rnn.pack_sequence(target, enforce_sorted=False)

    criterion = L1Loss()
    pit_criterion = ORPIT(criterion)
    loss, indices = pit_criterion(input, target)

    print(loss)
    print(indices)

def _test_sinkpit():
    random.seed(111)
    torch.manual_seed(111)

    batch_size, C, T = 4, 3, 1024
    input = torch.randint(2, (batch_size, C, T), dtype=torch.float)
    target = torch.randint(2, (batch_size, C, T), dtype=torch.float)

    print('-'*10, "Negative SI-SDR (PIT)", '-'*10)
    criterion = NegSISDR()
    pit_criterion = PIT(criterion, n_sources=C)
    loss, pattern = pit_criterion(input, target)

    print(loss)
    print(pattern)
    print()

    print('-'*10, "Negative SI-SDR", '-'*10)
    criterion = NegSISDR()
    pit_criterion = SinkPIT(criterion, n_sources=C, coldness=1)
    loss, pattern = pit_criterion(input, target, batch_mean=False)

    print(loss)
    print(pattern)
    print()

    print('-'*10, "SI-SDR", '-'*10)
    criterion = SISDR()
    pit_criterion = SinkPIT(criterion, n_sources=C, coldness=1)
    loss, pattern = pit_criterion(input, target, batch_mean=False)

    print(loss)
    print(pattern)
    print()

if __name__ == '__main__':
    import random

    from criterion.sdr import SISDR, NegSISDR
    from criterion.distance import L1Loss

    print('='*10, "Permutation invariant training", '='*10)
    _test_pit()
    print()

    print('='*10, "One-and-Rest permutation invariant training", '='*10)
    _test_orpit()
    print()

    print('='*10, "SinkPIT", '='*10)
    _test_sinkpit()
