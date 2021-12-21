import torch
import torch.nn as nn

"""
Feature-wise Linear Modulation
    Reference: "FiLM: Visual Reasoning with a General Conditioning Layer"
    See https://arxiv.org/abs/1709.07871
"""

class FiLM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, num_features, *)
            gamma (batch_size, num_features)
            beta (batch_size, num_features)
        Returns:
            output (batch_size, num_features, *)
        """
        n_dims = input.dim()
        expand_dims = (1,) * (n_dims - 2)
        dims = gamma.size() + expand_dims

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta

class FiLM1d(FiLM):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, num_features, T)
            gamma (batch_size, num_features)
            beta (batch_size, num_features)
        Returns:
            output (batch_size, num_features, T)
        """
        dims = gamma.size() + (1,)

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta

class FiLM2d(FiLM):
    def __init__(self):
        super().__init__()

    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, num_features, H, W)
            gamma (batch_size, num_features)
            beta (batch_size, num_features)
        Returns:
            output (batch_size, num_features, H, W)
        """
        dims = gamma.size() + (1, 1)

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta

class FiLM3d(FiLM):
    def __init__(self):
        super().__init__()

    def forward(self, input, gamma, beta):
        """
        Args:
            input (batch_size, num_features, H, W, D)
            gamma (batch_size, num_features)
            beta (batch_size, num_features)
        Returns:
            output (batch_size, num_features, H, W, D)
        """
        dims = gamma.size() + (1, 1, 1)

        gamma = gamma.view(*dims)
        beta = beta.view(*dims)

        return gamma * input + beta

def _test_film1d():
    torch.manual_seed(111)

    batch_size, in_channels = 4, 3

    print("-"*10, "FiLM1d", "-"*10)

    T = 5
    input = torch.randn((batch_size, in_channels, T), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, in_channels), dtype=torch.float), torch.randn((batch_size, in_channels), dtype=torch.float)
    model = FiLM1d()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())

def _test_film2d():
    torch.manual_seed(111)

    batch_size, in_channels = 4, 3

    print("-"*10, "FiLM2d", "-"*10)

    H, W = 5, 6
    input = torch.randn((batch_size, in_channels, H, W), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, in_channels), dtype=torch.float), torch.randn((batch_size, in_channels), dtype=torch.float)
    model = FiLM2d()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())

def _test_film3d():
    torch.manual_seed(111)

    batch_size, in_channels = 4, 3

    print("-"*10, "FiLM3d", "-"*10)

    H, W, D = 5, 6, 7
    input = torch.randn((batch_size, in_channels, H, W, D), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, in_channels), dtype=torch.float), torch.randn((batch_size, in_channels), dtype=torch.float)
    model = FiLM3d()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())

def _test_film():
    torch.manual_seed(111)

    batch_size, in_channels = 4, 3

    print("-"*10, "FiLM (any dimension is acceptable)", "-"*10)

    T = 5
    input = torch.randn((batch_size, in_channels, T), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, in_channels), dtype=torch.float), torch.randn((batch_size, in_channels), dtype=torch.float)
    model = FiLM()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())
    print()

    H, W, D = 5, 6, 7
    input = torch.randn((batch_size, in_channels, H, W, D), dtype=torch.float)
    gamma, beta = torch.randn((batch_size, in_channels), dtype=torch.float), torch.randn((batch_size, in_channels), dtype=torch.float)
    model = FiLM()
    output = model(input, gamma, beta)

    print(model)
    print(input.size(), output.size())
    print()

if __name__ == '__main__':
    _test_film1d()
    print()

    _test_film2d()
    print()

    _test_film3d()
    print()

    _test_film()