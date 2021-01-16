import itertools
import torch

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
        loss = criterion(input, target[:,pattern], batch_mean=False)
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

    def __call__(self, input, target, n_sources=None, batch_mean=True):
        """
        Args:
            input (batch_size, 2, *)
            target (batch_size, max(n_sources), *): `n_sources` is different for every input.
            n_sources (batch_size,): The number of sources per input. If `None`, the number is regarded same among the batch.
        Returns:
            loss (batch_size,): minimum loss for each data
            indices (batch_size,): most possible indices
        """
        assert input.size(1) == 2, "input.size() is expected (batch_size, 2, *), but given {}".format(input.size())

        criterion = self.criterion

        if n_sources is None:
            # Batch process
            n_sources = target.size(1)
        
            input_one, input_rest = input[:, 0], input[:, 1] # (batch_size, *), (batch_size, *)

            possible_loss = None
        
            for idx in range(n_sources):
                mask_one = torch.zeros_like(target)
                mask_one[:,idx] = 1.0
                mask_rest = torch.ones_like(target) - mask_one
                target_one = torch.sum(mask_one * target, dim=1)
                target_rest = torch.sum(mask_rest * target, dim=1)

                loss_one = criterion(input_one, target_one, batch_mean=False)
                loss_rest = criterion(input_rest, target_rest, batch_mean=False)
                loss = loss_one + loss_rest / (n_sources - 1)

                if possible_loss is None:
                    possible_loss = loss.unsqueeze(dim=1)
                else:
                    possible_loss = torch.cat([possible_loss, loss.unsqueeze(dim=1)], dim=1)
            
            # possible_loss (batch_size, n_sources)
            if criterion.maximize:
                batch_loss, batch_indices = torch.max(possible_loss, dim=1) # loss (batch_size,), batch_indices (batch_size,)
            else:
                batch_loss, batch_indices = torch.min(possible_loss, dim=1) # loss (batch_size,), batch_indices (batch_size,)
        else:
            print(n_sources)
            batch_size = input.size(0)
            batch_loss, indices = None, None

            for batch_idx in range(batch_size):
                _n_sources = n_sources[batch_idx] # <int>
                _input, _target = input[batch_idx: batch_idx+1], target[batch_idx: batch_idx+1, : _n_sources] # (1, 2, *), (1, n_sources, *)
                input_one, input_rest = _input[:, 0], _input[:, 1] # (1, *), (1, *)

                possible_loss = None
        
                for idx in range(_n_sources):
                    mask_one = torch.zeros_like(_target)
                    mask_one[:,idx] = 1.0
                    mask_rest = torch.ones_like(_target) - mask_one
                    target_one = torch.sum(mask_one * _target, dim=1) # (1, *)
                    target_rest = torch.sum(mask_rest * _target, dim=1) # (1, *)
                    
                    loss_one = criterion(input_one, target_one, batch_mean=False)
                    loss_rest = criterion(input_rest, target_rest, batch_mean=False)
                    loss = loss_one + loss_rest / (_n_sources - 1)

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


if __name__ == '__main__':
    import torch
    from criterion.sdr import SISDR
    from criterion.distance import L1Loss

    torch.manual_seed(111)

    batch_size, C, T = 4, 2, 1024
    
    criterion = SISDR()
    pit_criterion = PIT1d(criterion, n_sources=C)
    
    input = torch.randint(2, (batch_size, C, T), dtype=torch.float)
    target = torch.randint(2, (batch_size, C, T), dtype=torch.float)
    loss, pattern = pit_criterion(input, target)
    
    print(loss)
    print(pattern)
    print()

    batch_size, C, T = 4, 3, 5

    input = torch.randint(5, (batch_size, C, T), dtype=torch.float)
    target = torch.randint(5, (batch_size, C, T), dtype=torch.float)

    target[0] = input[0, torch.Tensor([2,1,0]).long()]
    target[1] = input[1, torch.Tensor([0,1,2]).long()]
    target[2] = input[2, torch.Tensor([0,2,1]).long()]
    target[3] = input[3, torch.Tensor([1,0,2]).long()]

    input_one = input[:, 0:1]
    input_rest = input[:, 1:].sum(dim=1, keepdim=True)
    input = torch.cat([input_one, input_rest], dim=1)

    criterion = L1Loss()
    pit_criterion = ORPIT(criterion)
    loss, indices = pit_criterion(input, target)
    
    print(loss)
    print(indices)
    print()

    input = torch.randint(5, (batch_size, C, T), dtype=torch.float)
    target = torch.randint(5, (batch_size, C, T), dtype=torch.float)
    n_sources = torch.Tensor([3,2,3,2]).long()
    
    target[0] = input[0, torch.Tensor([2,1,0]).long()]
    target[1] = input[1, torch.Tensor([0,1,2]).long()]
    target[2] = input[2, torch.Tensor([0,2,1]).long()]
    target[3] = input[3, torch.Tensor([1,0,2]).long()]

    input[1,2] = 0
    input[3,2] = 0
    target[1,2] = 0
    target[3,2] = 0

    input_one = input[:, 0:1]
    input_rest = input[:, 1:].sum(dim=1, keepdim=True)
    input = torch.cat([input_one, input_rest], dim=1)

    criterion = SISDR()
    pit_criterion = ORPIT(criterion)
    loss, indices = pit_criterion(input, target, n_sources=n_sources)

    print(loss)
    print(indices)
    print()