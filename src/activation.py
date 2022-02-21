import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-12

class ConcatenatedReLU(nn.Module):
    def __init__(self, dim=1):
        super().__init__()

        self.dim = dim

    def forward(self, input):
        positive, negative = F.relu(input), F.relu(-input)
        output = torch.cat([positive, negative], dim=self.dim)

        return output

"""
    For complex input
"""
class ModReLU1d(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features
        self.bias = nn.Parameter(torch.Tensor((num_features,)), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: Tensor with shape of
                (batch_size, num_features, T) if complex
                (batch_size, num_features, T, 2) otherwise
        Returns:
            output <torch.Tensor>: Tensor with shape of
                (batch_size, num_features, T) if complex
                (batch_size, num_features, T, 2) otherwise
        """
        is_complex = torch.is_complex(input)

        if not is_complex:
            input = torch.view_as_complex(input)

        magnitude = torch.abs(input)
        angle = torch.angle(input)
        magnitude = F.relu(magnitude + self.bias.unsqueeze(dim=-1))
        output = magnitude * torch.exp(1j * angle)

        if not is_complex:
            output = torch.view_as_real(output)

        return output

class ModReLU2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.num_features = num_features
        self.bias = nn.Parameter(torch.Tensor((num_features,)), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        self.bias.data.zero_()

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: Tensor with shape of
                (batch_size, num_features, height, width) if complex
                (batch_size, num_features, height, width, 2) otherwise
        Returns:
            output <torch.Tensor>: Tensor with shape of
                (batch_size, num_features, height, width) if complex
                (batch_size, num_features, height, width, 2) otherwise
        """
        is_complex = torch.is_complex(input)

        if not is_complex:
            input = torch.view_as_complex(input)

        magnitude = torch.abs(input)
        angle = torch.angle(input)
        magnitude = F.relu(magnitude + self.bias.unsqueeze(dim=-1).unsqueeze(dim=-1))
        output = magnitude * torch.exp(1j * angle)

        if not is_complex:
            output = torch.view_as_real(output)

        return output

"""
    See "Deep Complex Networks"
    https://arxiv.org/abs/1705.09792
"""

class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (*)
        Returns:
            output <torch.Tensor>: (*)
        """
        is_complex = torch.is_complex(input)

        if not is_complex:
            input = torch.view_as_complex(input)

        real, imag = input.real, input.imag
        real, imag = F.relu(real), F.relu(imag)

        output = torch.complex(real, imag)

        if not is_complex:
            output = torch.view_as_real(output)

        return output
"""
    z ReLU
    See "On complex valued convolutional neural networks" or "Deep Complex Networks"
    https://arxiv.org/abs/1602.09046
    https://arxiv.org/abs/1705.09792
"""

class ZReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: Complex tensor is acceptable. Shape: (*)
        Returns:
            output <torch.Tensor>: Real or complex tensor. Shape: (*)
        """
        is_complex = torch.is_complex(input)

        if not is_complex:
            input = torch.view_as_complex(input)

        real, imag = input.real, input.imag

        condition = torch.logical_and(real > 0, imag > 0)
        real = torch.where(condition, real, torch.zeros_like(real))
        imag = torch.where(condition, imag, torch.zeros_like(imag))

        output = torch.complex(real, imag)

        if not is_complex:
            output = torch.view_as_real(output)

        return output

def _test_mod_relu():
    batch_size, num_features, T = 4, 3, 8
    real, imag = torch.randn(batch_size, num_features, T), torch.randn(batch_size, num_features, T)

    input = torch.complex(real, imag)
    model = ModReLU1d(num_features)
    output_as_complex = model(input)

    print(model)
    print(input.size(), output_as_complex.size())
    print()

    input = torch.view_as_real(input)
    output_as_real = model(input)

    print(model)
    print(input.size(), output_as_real.size())

    print(torch.all(output_as_complex == torch.view_as_complex(output_as_real)))

def _test_complex_relu():
    batch_size, num_features, T = 4, 3, 8
    real, imag = torch.randn(batch_size, num_features, T), torch.randn(batch_size, num_features, T)

    input = torch.complex(real, imag)
    model = ComplexReLU()
    output_as_complex = model(input)

    print(model)
    print(input.size(), output_as_complex.size())
    print()

    input = torch.view_as_real(input)
    output_as_real = model(input)

    print(model)
    print(input.size(), output_as_real.size())

    print(torch.all(output_as_complex == torch.view_as_complex(output_as_real)))

def _test_zrelu():
    batch_size, num_features, T = 4, 3, 8
    real, imag = torch.randn(batch_size, num_features, T), torch.randn(batch_size, num_features, T)

    input = torch.complex(real, imag)
    model = ZReLU()
    output_as_complex = model(input)

    print(model)
    print(input.size(), output_as_complex.size())
    print()

    input = torch.view_as_real(input)
    output_as_real = model(input)

    print(model)
    print(input.size(), output_as_real.size())

    print(torch.all(output_as_complex == torch.view_as_complex(output_as_real)))

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "ModReLU", "="*10)
    _test_mod_relu()
    print()

    print("="*10, "ComplexReLU", "="*10)
    _test_complex_relu()
    print()

    print("="*10, "ZReLU", "="*10)
    _test_zrelu()
    print()