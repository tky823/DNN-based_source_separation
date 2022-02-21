import torch.nn as nn

class SquaredError(nn.Module):
    def __init__(self, sum_dim=None, mean_dim=None):
        super().__init__()

        if type(sum_dim) is int:
            sum_dim = (sum_dim,)

        if type(mean_dim) is int:
            mean_dim = (mean_dim,)

        self.sum_dim, self.mean_dim = sum_dim, mean_dim

        dims = self.reduction_dims

        if len(set(dims)) != len(dims):
            raise ValueError("`sum_dim` and `mean_dim` have some same values.")

    def forward(self, input, target, batch_mean=True):
        sum_dim, mean_dim = self.sum_dim, self.mean_dim

        loss = (input - target)**2

        n_dims = loss.dim()
        dims = (
            n_dims + dim if dim < 0 else dim for dim in self.reduction_dims
        )
        dims = sorted(dims)[::-1]

        if sum_dim is not None:
            loss = loss.sum(dim=sum_dim, keepdim=True)

        if mean_dim is not None:
            loss = loss.mean(dim=mean_dim, keepdim=True)

        for dim in dims:
            loss = loss.squeeze(dim=dim)

        if batch_mean:
            loss = loss.mean(dim=0)

        return loss

    @property
    def maximize(self):
        return False

    @property
    def reduction_dims(self):
        sum_dim, mean_dim = self.sum_dim, self.mean_dim
        _dims = ()

        if sum_dim is not None:
            _dims = _dims + sum_dim

        if mean_dim is not None:
            _dims = _dims + mean_dim

        return _dims