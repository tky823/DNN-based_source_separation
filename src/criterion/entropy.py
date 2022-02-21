import torch
import torch.nn as nn

EPS = 1e-12

class BinaryCrossEntropy(nn.Module):
    def __init__(self, reduction="mean", eps=EPS):
        super().__init__()

        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, *)
            target (batch_size, *)
        Returns:
            loss () or (batch_size,)
        """
        reduction = self.reduction
        eps = self.eps

        loss = - target * torch.log(input + eps) - (1 - target) * torch.log(1 - input + eps)

        n_dims = loss.dim()
        dim = tuple(range(1, n_dims))

        if reduction == "mean":
            loss = loss.mean(dim=dim)
        elif reduction == "sum":
            loss = loss.sum(dim=dim)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

class CategoricalCrossEntropy(nn.Module):
    def __init__(self, class_dim=1, reduction="mean", eps=EPS):
        super().__init__()

        self.class_dim = class_dim
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target, batch_mean=True):
        """
        Args:
            input (batch_size, num_classes, *)
            target (batch_size, num_classes, *)
        Returns:
            loss () or (batch_size,)
        """
        reduction = self.reduction
        eps = self.eps

        loss = - target * torch.log(input + eps)
        loss = loss.sum(dim=self.class_dim)

        n_dims = loss.dim()
        dim = tuple(range(1, n_dims))

        if reduction == "mean":
            loss = loss.mean(dim=dim)
        elif reduction == "sum":
            loss = loss.sum(dim=dim)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss