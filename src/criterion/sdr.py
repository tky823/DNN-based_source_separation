import torch
import torch.nn as nn

EPS=1e-12

"""
    Scale-invariant-SDR (source-to-distortion ratio)
    See "SDR - half-baked or well done?"
    https://arxiv.org/abs/1811.02508
"""

def sisdr(input, target, eps=EPS):
    """
    Scale-invariant-SDR (source-to-distortion ratio)
    Args:
        input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
        target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
    Returns:
        loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
    """
    n_dims = input.dim()
    
    assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)

    alpha = torch.sum(input * target, dim=n_dims-1, keepdim=True) / (torch.sum(target**2, dim=n_dims-1, keepdim=True) + eps)
    loss = (torch.sum((alpha * target)**2, dim=n_dims-1) + eps) / (torch.sum((alpha * target - input)**2, dim=n_dims-1) + eps)
    loss = 10 * torch.log10(loss)

    return loss

class SISDR(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
        
        self.maximize = True
        self.eps = eps
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            iinput (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
        """
        n_dims = input.dim()
        
        assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dim)
        
        loss = sisdr(input, target, eps=self.eps)
        
        if n_dims == 3:
            loss = loss.mean(dim=1)
        elif n_dims == 4:
            loss = loss.mean(dim=(1,2))
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
        
class NegSISDR(nn.Module):
    def __init__(self, eps=EPS):
        super().__init__()
        
        self.maximize = False
        self.eps = eps
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, T) or (batch_size, C, T)
            target (batch_size, T) or (batch_size, C, T)
        Returns:
            loss (batch_size,)
        """
        n_dims = input.dim()
        
        assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dim)
        
        loss = - sisdr(input, target, eps=self.eps)
        
        if n_dims == 3:
            loss = loss.mean(dim=1)
        elif n_dims == 4:
            loss = loss.mean(dim=(1,2))
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
