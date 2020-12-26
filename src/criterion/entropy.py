import torch
import torch.nn as nn

EPS=1e-12

class BinaryCrossEntropy(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
    
    def forward(input, target, batch_mean=True):
        """
        Args:
            input (batch_size, C, T)
            target (batch_size, C, T)
        Returns:
            loss () or (batch_size, )
        """
        loss = - target * torch.log(input + EPS) - (1 - target) * torch.log(1 - input + EPS)
        loss = loss.squeeze(dim=1).mean(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss

class CrossEntropy(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
    
    def forward(input, target, batch_mean=True):
        """
        Args:
            input (batch_size, C, T)
            target (batch_size, C, T)
        Returns:
            loss () or (batch_size, )
        """
        loss = - target * torch.log(input + EPS) - (1 - target) * torch.log(1 - input + EPS)
        loss = loss.squeeze(dim=1).mean(dim=1)
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss