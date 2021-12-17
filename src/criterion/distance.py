import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class L1Loss(nn.Module):
    def __init__(self, dim=1, reduction='mean'):
        super().__init__()

        self.dim = dim
        self.reduction = reduction

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input: (batch_size, *)
            target: (batch_size, *)
        Returns:
            output: () or (batch_size, )
        """
        loss = torch.abs(input - target) # (batch_size, *)
        loss = torch.sum(loss, dim=self.dim)

        n_dims = loss.dim()
        dim = tuple(range(1, n_dims))

        if n_dims > 1:
            if self.reduction == 'mean':
                loss = loss.mean(dim=dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dim)
            else:
                raise ValueError("Invalid reduction type")

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

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

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input: (batch_size, *)
            target: (batch_size, *)
        Returns:
            output: () or (batch_size, )
        """
        loss = torch.abs(input - target) # (batch_size, *)
        loss = torch.sum(loss**2, dim=self.dim)
        loss = torch.sqrt(loss)

        n_dims = loss.dim()
        if n_dims > 1:
            dim = tuple(range(1, n_dims))

            if self.reduction == 'mean':
                loss = loss.mean(dim=dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dim)
            else:
                raise ValueError("Invalid reduction type")

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

class L12Loss(nn.Module):
    def __init__(self, dim1=1, dim2=2, reduction='mean'):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()

        self.dim1, self.dim2 = dim1, dim2
        self.reduction = reduction

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

    def forward(self, input, target, batch_mean=False):
        loss = torch.abs(input - target) # (batch_size, *)
        loss = torch.sum(loss, dim=self.dim1, keepdim=True)
        loss = torch.sum(loss**2, dim=self.dim2, keepdim=True)
        loss = torch.sqrt(loss)

        n_dims = loss.dim()
        if n_dims > 1:
            dim = tuple(range(1, n_dims))

            if self.reduction == 'mean':
                loss = loss.mean(dim=dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dim)
            else:
                raise ValueError("Invalid reduction type")

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

class L21Loss(nn.Module):
    def __init__(self, dim1=1, dim2=2, reduction='mean'):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()

        self.dim1, self.dim2 = dim1, dim2
        self.reduction = reduction

        if not reduction in ['mean', 'sum']:
            raise ValueError("Invalid reduction type")

    def forward(self, input, target, batch_mean=False):
        loss = torch.abs(input - target) # (batch_size, *)
        loss = torch.sum(loss**2, dim=self.dim2, keepdim=True)
        loss = torch.sqrt(loss)
        loss = torch.sum(loss, dim=self.dim1, keepdim=True)

        n_dims = loss.dim()
        if n_dims > 1:
            dim = tuple(range(1, n_dims))

            if self.reduction == 'mean':
                loss = loss.mean(dim=dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dim)
            else:
                raise ValueError("Invalid reduction type")

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

class SquaredError(nn.Module):
    def __init__(self, reduction=None, reduction_dim=None):
        super().__init__()

        self.reduction = reduction
        self.reduction_dim = reduction_dim

    def forward(self, input, target, batch_mean=True):
        loss = (input - target)**2

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
            else:
                raise NotImplementedError("Not support self.reduction={}.".format(self.reduction))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

class MeanAbsoluteError(nn.Module):
    def __init__(self, dim=1, reduction=None):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()

        self.dim = dim
        self.reduction = reduction

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *):
            target (batch_size, *):
        """
        loss = torch.abs(input - target) # (batch_size, *)
        loss = torch.mean(loss, dim=self.dim)

        n_dims = loss.dim()

        if self.reduction:
            dim = tuple(range(1, n_dims))
            if self.reduction == 'mean':
                loss = loss.mean(dim=dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dim)
            else:
                raise NotImplementedError("Not support self.reduction={}.".format(self.reduction))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

class MeanSquaredError(nn.Module):
    def __init__(self, dim=1, reduction=None):
        """
        Args:
            dim <int> or <tuple<int>>
        """
        super().__init__()

        self.dim = dim
        self.reduction = reduction

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *):
            target (batch_size, *):
        """
        loss = (input - target)**2 # (batch_size, *)
        loss = torch.mean(loss, dim=self.dim)

        n_dims = loss.dim()

        if self.reduction:
            dim = tuple(range(1, n_dims))
            if self.reduction == 'mean':
                loss = loss.mean(dim=dim)
            elif self.reduction == 'sum':
                loss = loss.sum(dim=dim)
            else:
                raise NotImplementedError("Not support self.reduction={}.".format(self.reduction))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

class CosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, maximize=False, eps=EPS):
        super().__init__()

        self.dim = dim
        self.maximize = maximize
        self.eps = eps

    def forward(self, input, target, batch_mean=False):
        loss = F.cosine_similarity(input, target, dim=self.dim, eps=self.eps)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class NegCosineSimilarityLoss(nn.Module):
    def __init__(self, dim=1, maximize=True, eps=EPS):
        super().__init__()

        self.dim = dim
        self.maximize = maximize
        self.eps = eps

    def forward(self, input, target, batch_mean=False):
        loss = - F.cosine_similarity(input, target, dim=self.dim, eps=self.eps)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

def _test_l1loss():
    batch_size, C, T = 2, 4, 6

    input = torch.randn(batch_size, C, T)
    target = torch.randn(batch_size, C, T)

    criterion = L1Loss(dim=2)
    loss = criterion(input, target)
    print(loss)

def _test_l12loss():
    batch_size, C, H, W = 2, 4, 6, 10

    input = torch.randn(batch_size, C, H, W)
    target = torch.randn(batch_size, C, H, W)

    criterion = L12Loss(dim1=2, dim2=3)
    loss = criterion(input, target)
    print(loss)

def _test_l21loss():
    batch_size, C, H, W = 2, 4, 6, 10

    input = torch.randn(batch_size, C, H, W)
    target = torch.randn(batch_size, C, H, W)

    criterion = L21Loss(dim1=2, dim2=3)
    loss = criterion(input, target)
    print(loss)

if __name__ == '__main__':
    torch.manual_seed(111)

    _test_l1loss()
    print()

    _test_l12loss()
    print()

    _test_l21loss()