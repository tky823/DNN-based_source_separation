import torch
import torch.nn as nn

EPS=1e-12

class L1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        
        self.reduction = reduction
        
        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")
        
        self.maximize = False
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *):
            target (batch_size, *):
        """
        n_dim = input.dim()
        dim = tuple(range(1,n_dim))
        
        loss = torch.abs(input - target) # (batch_size, *)
        
        if self.reduction == 'mean':
            loss = loss.mean(dim=dim)
        elif self.reduction == 'sum':
            loss = loss.sum(dim=dim)
        else:
            raise ValueError("Invalid reduction type")
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss

class L2Loss(nn.Module):
    def __init__(self, dim=1, reduction='mean'):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()
        
        self.dim = dim
        self.reduction = reduction
        
        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")
        
        self.maximize = False
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *):
            target (batch_size, *):
        """
        loss = torch.abs(input - target) # (batch_size, *)
        loss = torch.sum(loss**2, dim=self.dim)
        loss = torch.sqrt(loss)
        
        n_dim = loss.dim()
        dim = tuple(range(1,n_dim))
        
        if self.reduction == 'mean':
            loss = loss.mean(dim=dim)
        elif self.reduction == 'sum':
            loss = loss.sum(dim=dim)
        else:
            raise ValueError("Invalid reduction type")
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss

class MeanSquaredError(nn.Module):
    def __init__(self, dim=1):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()
        
        self.dim = dim
        
        self.maximize = False
    
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *):
            target (batch_size, *):
        """
        loss = (input - target)**2 # (batch_size, *)
        loss = torch.mean(loss, dim=self.dim)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
