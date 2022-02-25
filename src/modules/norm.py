import torch
import torch.nn as nn

EPS = 1e-12

"""
    Global layer normalization
    See "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
    https://arxiv.org/abs/1809.07454
"""
class GlobalLayerNorm(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, *)
        Returns:
            output (batch_size, C, *)
        """
        output = self.norm(input)

        return output

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)

"""
    Cumulative layer normalization
    See "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation"
    https://arxiv.org/abs/1809.07454
"""
class CumulativeLayerNorm1d(nn.Module):
    def __init__(self, num_features, eps=EPS):
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.fill_(0)

    def forward(self, input):
        """
        Args:
            input (batch_size, C, T) or (batch_size, C, S, chunk_size):
        Returns:
            output (batch_size, C, T) or (batch_size, C, S, chunk_size): same shape as the input
        """
        eps = self.eps

        n_dims = input.dim()

        if n_dims == 3:
            batch_size, C, T = input.size()
        elif n_dims == 4:
            batch_size, C, S, chunk_size = input.size()
            T = S * chunk_size
            input = input.view(batch_size, C, T)
        else:
            raise ValueError("Only support 3D or 4D input, but given {}D".format(input.dim()))

        step_sum = torch.sum(input, dim=1) # (batch_size, T)
        step_squared_sum = torch.sum(input**2, dim=1) # (batch_size, T)
        cum_sum = torch.cumsum(step_sum, dim=1) # (batch_size, T)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1) # (batch_size, T)

        cum_num = torch.arange(C, C * (T + 1), C, dtype=torch.float) # (T, ): [C, 2*C, ..., T*C]
        cum_mean = cum_sum / cum_num # (batch_size, T)
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean**2

        cum_mean, cum_var = cum_mean.unsqueeze(dim=1), cum_var.unsqueeze(dim=1)

        output = (input - cum_mean) / (torch.sqrt(cum_var) + eps) * self.gamma + self.beta

        if n_dims == 4:
            output = output.view(batch_size, C, S, chunk_size)

        return output

    def __repr__(self):
        s = '{}'.format(self.__class__.__name__)
        s += '({num_features}, eps={eps})'

        return s.format(**self.__dict__)

"""
TODO: Virtual batch normalization
"""

if __name__ == '__main__':
    batch_size, C, T = 2, 3, 5
    causal = True

    norm = GlobalLayerNorm(C)
    print(norm)

    input = torch.arange(batch_size*C*T, dtype=torch.float).view(batch_size, C, T)
    output = norm(input)
    print(input)
    print(output)