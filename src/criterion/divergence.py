import torch
import torch.nn as nn

EPS = 1e-12

def kl_divergence(input, target, eps=EPS):
    """
    Args:
        input <torch.Tensor>: (*)
        target <torch.Tensor>: (*)
    Returns:
        loss <torch.Tensor>: (*)
    """
    ratio = (target + eps) / (input + eps)
    loss = target * torch.log(ratio)
    loss = loss.sum(dim=0)

    return loss

def is_divergence(input, target, eps=EPS):
    """
    Args:
        input <torch.Tensor>: (*)
        target <torch.Tensor>: (*)
    Returns:
        loss <torch.Tensor>: (*)
    """
    ratio = (target + eps) / (input + eps)
    loss = ratio - torch.log(ratio) - 1

    return loss

def generalized_kl_divergence(input, target, eps=EPS):
    """
    Args:
        input <torch.Tensor>: (*)
        target <torch.Tensor>: (*)
    Returns:
        loss <torch.Tensor>: (*)
    """
    ratio = (target + eps) / (input + eps)
    loss = target * torch.log(ratio) + input - target

    return loss

def beta_divergence(input, target, beta=2):
    """
    Beta divergence
    Args:
        input <torch.Tensor>: (*)
        target <torch.Tensor>: (*)
    Returns:
        loss <torch.Tensor>: (*)
    """
    beta_minus1 = beta - 1

    assert beta != 0, "Use is_divergence instead."
    assert beta_minus1 != 0, "Use generalized_kl_divergence instead."

    loss = target * (target**beta_minus1 - input**beta_minus1) / beta_minus1 - (target**beta - input**beta) / beta

    return loss

class KLdivergence(nn.Module):
    def __init__(self, reduction='sum', eps=EPS):
        super().__init__()

        self.reduction = reduction
        self.maximize = False
        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, C, *)
            target (batch_size, C, *)
        Returns:
            loss () or (batch_size, )
        """
        reduction = self.reduction
        n_dims = input.dim()

        permuted_dims = [1, 0] + list(range(2, n_dims))
        input, target = input.permute(*permuted_dims), target.permute(*permuted_dims)
        loss = kl_divergence(input, target, eps=self.eps)

        dims = tuple(range(1, n_dims - 1))

        if reduction == 'sum':
            loss = loss.sum(dim=dims)
        elif reduction == 'mean':
            loss = loss.mean(dim=dims)
        else:
            raise NotImplementedError("Not support {} for reduction".format(reduction))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class ISdivergence(nn.Module):
    def __init__(self, reduction='sum', eps=EPS):
        super().__init__()

        self.reduction = reduction
        self.maximize = False
        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *)
            target (batch_size, *)
        Returns:
            loss () or (batch_size, )
        """
        reduction = self.reduction
        n_dims = input.dim()

        loss = is_divergence(input, target, eps=self.eps)

        dims = tuple(range(1, n_dims))

        if reduction == 'sum':
            loss = loss.sum(dim=dims)
        elif reduction == 'mean':
            loss = loss.mean(dim=dims)
        else:
            raise NotImplementedError("Not support {} for reduction".format(reduction))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class GeneralizedKLdivergence(nn.Module):
    def __init__(self, reduction='sum', eps=EPS):
        super().__init__()

        self.reduction = reduction
        self.maximize = False
        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *)
            target (batch_size, *)
        Returns:
            loss () or (batch_size, )
        """
        reduction = self.reduction
        n_dims = input.dim()

        loss = generalized_kl_divergence(input, target, eps=self.eps)

        dims = tuple(range(1, n_dims))

        if reduction == 'sum':
            loss = loss.sum(dim=dims)
        elif reduction == 'mean':
            loss = loss.mean(dim=dims)
        else:
            raise NotImplementedError("Not support {} for reduction".format(reduction))

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

if __name__ =='__main__':
    torch.manual_seed(111)

    batch_size, C = 2, 3
    F_bin, T_bin = 3, 4

    input = torch.rand(batch_size, F_bin, T_bin, dtype=torch.float)
    target = torch.rand(batch_size, F_bin, T_bin, dtype=torch.float)
    criterion = ISdivergence()
    loss = criterion(input, target)

    print(loss)

    input = torch.rand(batch_size, F_bin, T_bin, dtype=torch.float)
    target = torch.rand(batch_size, F_bin, T_bin, dtype=torch.float)
    criterion = GeneralizedKLdivergence()
    loss = criterion(input, target)

    print(loss)

    input = torch.rand(batch_size, C, F_bin, T_bin, dtype=torch.float)
    target = torch.rand(batch_size, C, F_bin, T_bin, dtype=torch.float)
    criterion = KLdivergence()
    loss = criterion(input, target)

    print(loss)