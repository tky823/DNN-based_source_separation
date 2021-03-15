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
    possible_loss = None
    
    for idx in range(P):
        pattern = patterns[idx]
        loss = criterion(input, target[:, pattern], batch_mean=False)
        if possible_loss is None:
            possible_loss = loss.unsqueeze(dim=1)
        else:
            possible_loss = torch.cat([possible_loss, loss.unsqueeze(dim=1)], dim=1)
            
    # possible_loss (batch_size, P)
    if criterion.maximize:
        loss, indices = torch.max(possible_loss, dim=1) # loss (batch_size,), indices (batch_size,)
    else:
        loss, indices = torch.min(possible_loss, dim=1) # loss (batch_size,), indices (batch_size,)
    
    if batch_mean:
        loss = loss.mean(dim=0)
         
    return loss, patterns[indices]
        

class PIT:
    def __init__(self, criterion, n_sources):
        """
        Args:
            criterion <callable>: criterion is expected acceptable (input, target, batch_mean) when called.
        """
        self.criterion = criterion
        patterns = list(itertools.permutations(range(n_sources)))
        self.patterns = torch.Tensor(patterns).long()
        
    def __call__(self, input, target, batch_mean=True):
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

class ORPIT:
    """
    One-and-Rest permutation invariant training
    """
    def __init__(self, criterion):
        self.criterion = criterion
        patterns = list(itertools.permutations(range(2))) # 2 means 'one' and 'rest'
        self.patterns = torch.Tensor(patterns).long()

    def __call__(self, input, target, batch_mean=True):
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

        target, lens_unpacked = nn.utils.rnn.pad_packed_sequence(target, batch_first=True)

        # TODO: batch process

        batch_size = input.size(0)
        batch_loss, batch_indices = None, None

        for batch_idx in range(batch_size):
            n_sources = lens_unpacked[batch_idx] # <int>
            _input, _target = input[batch_idx: batch_idx+1], target[batch_idx: batch_idx+1, : n_sources] # (1, 2, *), (1, n_sources, *)
            input_one, input_rest = _input[:, 0], _input[:, 1] # (1, *), (1, *)

            possible_loss = None
    
            for idx in range(n_sources):
                mask_one = torch.zeros_like(_target)
                mask_one[:,idx] = 1.0
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
                
            if criterion.maximize:
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

def _test_pit():
    from criterion.sdr import SISDR

    torch.manual_seed(111)

    batch_size, C, T = 4, 2, 1024
    
    criterion = SISDR()
    pit_criterion = PIT1d(criterion, n_sources=C)
    
    input = torch.randint(2, (batch_size, C, T), dtype=torch.float)
    target = torch.randint(2, (batch_size, C, T), dtype=torch.float)
    loss, pattern = pit_criterion(input, target)
    
    print(loss)
    print(pattern)

def _test_orpit():
    from criterion.sdr import SISDR
    from criterion.distance import L1Loss
    
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

if __name__ == '__main__':
    print('='*10, "Permutation invariant training", '='*10)
    _test_pit()
    print()

    print('='*10, "One-and-Rest permutation invariant training", '='*10)
    _test_orpit()
    print()
