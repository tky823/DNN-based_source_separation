import itertools
import torch

"""
    Permutation invariant training
"""

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
            patterns (batch_size,): permutation indices
        """
        criterion = self.criterion
        patterns = self.patterns
        P = len(patterns)
        possible_loss = None
        
        for idx in range(P):
            pattern = patterns[idx]
            loss = self.criterion(input, target[:,pattern], batch_mean=False)
            if possible_loss is None:
                possible_loss = loss.unsqueeze(dim=1)
            else:
                possible_loss = torch.cat([possible_loss, loss.unsqueeze(dim=1)], dim=1)
                
        # possible_loss (batch_size, P)
        if self.criterion.maximize:
            loss, idx_max = torch.max(possible_loss, dim=1) # loss (batch_size, ), idx_min (batch_size, )s
        else:
            loss, idx_min = torch.min(possible_loss, dim=1) # loss (batch_size, ), idx_min (batch_size, )
        if batch_mean:
            loss = loss.mean(dim=0)
             
        return loss, patterns[idx_min]

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
