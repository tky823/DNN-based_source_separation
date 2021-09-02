import torch
import torch.nn as nn

EPS = 1e-12

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
    def __init__(self, reduction='mean', eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ['mean', 'sum', None]:
            raise ValueError("Invalid reduction type")
        
        self.eps = eps
        
    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
        """
        n_dims = input.dim()
        
        assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)
        
        loss = sisdr(input, target, eps=self.eps)
        
        if self.reduction:
            if n_dims == 3:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=1)
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=(1, 2))
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
    
    @property
    def maximize(self):
        return True
        
class NegSISDR(nn.Module):
    def __init__(self, reduction='mean', eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ['mean', 'sum', None]:
            raise ValueError("Invalid reduction type")
        
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
        
        assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)
        
        loss = - sisdr(input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=1)
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=(1, 2))
        
        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
    
    @property
    def maximize(self):
        return False

"""
    Weighted SDR (signal-to-distortion ratio)
    See "Phase-Aware Speech Enhancement with Deep Complex U-Net"
"""
def weighted_sdr(mixture, input, target, eps=EPS):
    """
    Args:
        mixture <torch.Tensor>: (*, 1, *, T)
        input <torch.Tensor>: (*, n_sources, *, T)
        target <torch.Tensor>: (*, n_sources, *, T)
    Returns:
        loss <torch.Tensor>: (*, n_sources, *)
    """
    mixture, input, target

    target_power = torch.sum(target**2, dim=-1)
    loss = torch.sum(target * input, dim=-1) / (target_power * torch.sum(input**2, dim=-1) + eps)

    residual_input, residual_target = mixture - input, mixture - target
    residual_target_power = torch.sum(residual_target**2, dim=-1)
    loss_residual = torch.sum(residual_target * residual_input, dim=-1) / (torch.sum(residual_target**2, dim=-1) * torch.sum(residual_input**2, dim=-1) + eps)

    rho = target_power / (target_power + residual_target_power + eps)

    loss = rho * loss + (1 - rho) * loss_residual

    return loss

class WeightedSDR(nn.Module):
    def __init__(self, reduction='mean', eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

        self.eps = eps
    
    def forward(self, mixture, input, target, batch_mean=True):
        """
        Args:
            mixture <torch.Tensor>: (batch_size, T) or (batch_size, 1, T), or (batch_size, 1, n_mics, T)
            input <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss <torch.Tensor>: (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics) if batch_mean=False
        """
        n_dims = mixture.dim()

        assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)

        loss = weighted_sdr(mixture, input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=1)
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=(1, 2))

        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
    
    @property
    def maximize(self):
        return True

class NegWeightedSDR(nn.Module):
    def __init__(self, reduction='mean', eps=EPS):
        super().__init__()

        self.reduction = reduction

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

        self.eps = eps
    
    def forward(self, mixture, input, target, batch_mean=True):
        """
        Args:
            mixture <torch.Tensor>: (batch_size, T) or (batch_size, 1, T), or (batch_size, 1, n_mics, T)
            input <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss <torch.Tensor>: (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics) if batch_mean=False
        """
        n_dims = mixture.dim()

        assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)

        loss = - weighted_sdr(mixture, input, target, eps=self.eps)

        if self.reduction:
            if n_dims == 3:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=1)
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=1)
            elif n_dims == 4:
                if self.reduction == 'mean':
                    loss = loss.mean(dim=(1, 2))
                elif self.reduction == 'sum':
                    loss = loss.sum(dim=(1, 2))

        if batch_mean:
            loss = loss.mean(dim=0)
        
        return loss
    
    @property
    def maximize(self):
        return False

def _test_sisdr():
    pass

def _test_weighted_sdr():
    print("-"*10, "weighted SDR", "-"*10)
    criterion = WeightedSDR()

    print(criterion.maximize)
    print()

    print("-"*10, "Negative weighted SDR", "-"*10)
    criterion = NegWeightedSDR()

    print(criterion.maximize)

if __name__ == '__main__':
    _test_sisdr()

    _test_weighted_sdr()