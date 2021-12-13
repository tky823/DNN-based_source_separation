import torch
import torch.nn as nn

EPS = 1e-12

def sdr(input, target, eps=EPS):
    """
    Source-to-distortion ratio (SDR)
    Args:
        input (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
        target (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
    Returns:
        loss (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics)
    """
    n_dims = input.dim()

    assert n_dims in [2, 3, 4], "Only 2D or 3D or 4D tensor is acceptable, but given {}D tensor.".format(n_dims)

    loss = (torch.sum(target**2, dim=n_dims-1) + eps) / (torch.sum((target - input)**2, dim=n_dims-1) + eps)
    loss = 10 * torch.log10(loss)

    return loss

class SDR(nn.Module):
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

        loss = sdr(input, target, eps=self.eps)

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

class NegSDR(nn.Module):
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

        loss = - sdr(input, target, eps=self.eps)

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

class ClippedSISDR(nn.Module):
    def __init__(self, max=None, reduction='mean', eps=EPS):
        super().__init__()

        self.max = max
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
        loss = torch.clamp(loss, max=self.max)

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

class ClippedNegSISDR(nn.Module):
    def __init__(self, min=None, reduction='mean', eps=EPS):
        super().__init__()

        self.min = min
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
        loss = torch.clamp(loss, min=self.min)

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
def weighted_sdr(input, target, source_dim=1, eps=EPS):
    """
    Args:
        input <torch.Tensor>: (*, n_sources, *, T)
        target <torch.Tensor>: (*, n_sources, *, T)
    Returns:
        loss <torch.Tensor>: (*, n_sources, *)
    """
    mixture = target.sum(dim=source_dim, keepdim=True) # (*, 1, *, T)
    
    target_power = torch.sum(target**2, dim=-1)
    loss = (torch.sum(target * input, dim=-1) + eps) / (torch.linalg.vector_norm(target, dim=-1) * torch.linalg.vector_norm(input, dim=-1) + eps)

    residual_input, residual_target = mixture - input, mixture - target
    residual_target_power = torch.sum(residual_target**2, dim=-1)
    loss_residual = (torch.sum(residual_target * residual_input, dim=-1) + eps) / (torch.linalg.vector_norm(residual_target, dim=-1) * torch.linalg.vector_norm(residual_input, dim=-1) + eps)

    rho = (target_power + eps) / (target_power + residual_target_power + eps)
    loss = rho * loss + (1 - rho) * loss_residual

    return loss

class WeightedSDR(nn.Module):
    def __init__(self, source_dim=1, reduction='mean', reduction_dim=None, eps=EPS):
        super().__init__()

        self.source_dim = source_dim
        self.reduction = reduction
        self.reduction_dim = reduction_dim

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss <torch.Tensor>: (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics) if batch_mean=False
        """
        loss = weighted_sdr(input, target, source_dim=self.source_dim, eps=self.eps)

        if self.reduction:
            if self.reduction_dim:
                reduction_dim = self.reduction_dim
            else:
                n_dims = loss.dim()
                reduction_dim = tuple(range(1, n_dims))

            if self.reduction == 'mean':
                loss = loss.mean(dim=reduction_dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=reduction_dim)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return True

class NegWeightedSDR(nn.Module):
    def __init__(self, source_dim=1, reduction='mean', reduction_dim=None, eps=EPS):
        super().__init__()

        self.source_dim = source_dim
        self.reduction = reduction
        self.reduction_dim = reduction_dim

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T), or (batch_size, n_sources, n_mics, T)
            target <torch.Tensor>: (batch_size, T) or (batch_size, n_sources, T) or (batch_size, n_sources, n_mics, T)
        Returns:
            loss <torch.Tensor>: (batch_size,) or (batch_size, n_sources) or (batch_size, n_sources, n_mics) if batch_mean=False
        """
        loss = - weighted_sdr(input, target, source_dim=self.source_dim, eps=self.eps)

        if self.reduction:
            if self.reduction_dim:
                reduction_dim = self.reduction_dim
            else:
                n_dims = loss.dim()
                reduction_dim = tuple(range(1, n_dims))

            if self.reduction == 'mean':
                loss = loss.mean(dim=reduction_dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=reduction_dim)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

def _test_sisdr():
    pass

def _test_weighted_sdr():
    batch_size = 3
    n_sources = 4
    in_channels, T = 2, 32

    print("-"*10, "weighted SDR", "-"*10)
    input = torch.randn(batch_size, n_sources, in_channels, T)
    target = input.clone()

    criterion = WeightedSDR()
    loss = criterion(input, target)

    print(criterion.maximize)
    print(loss)
    print()

    print("-"*10, "Negative weighted SDR", "-"*10)
    input = torch.randn(batch_size, n_sources, in_channels, T)
    target = torch.randn(batch_size, n_sources, in_channels, T)

    criterion = NegWeightedSDR()
    loss = criterion(input, target)

    print(criterion.maximize)
    print(loss)

if __name__ == '__main__':
    print("="*10, "SI-SDR", "="*10)
    _test_sisdr()

    print("="*10, "Weighted SDR", "="*10)
    _test_weighted_sdr()