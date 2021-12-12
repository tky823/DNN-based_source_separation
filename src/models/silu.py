import torch
import torch.nn as nn

"""
Sigmoid Linear Units
    Reference: "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning"
    See https://arxiv.org/abs/1702.03118
"""

class SiLU1d(nn.Module):
    """
    Sigmoid Linear Units for 1D inputs
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        """
        Args:
            in_channels <int>
            out_channels <int>
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T)
        Returns:
            output (batch_size, out_channels, T)
        """
        x = self.map(input)
        x_gate = torch.sigmoid(x)

        output = x * x_gate

        return output

class SiLU2d(nn.Module):
    """
    Sigmoid Linear Units for 2D inputs
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1)):
        """
        Args:
            in_channels <int>
            out_channels <int>
        """
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output (batch_size, out_channels, H, W)
        """
        x = self.map(input)
        x_gate = torch.sigmoid(x)

        output = x * x_gate

        return output

def _test_silu1d():
    torch.manual_seed(111)

    batch_size = 4
    in_channels, out_channels = 3, 16
    kernel_size, stride = 3, 2
    T = 128

    input = torch.rand(batch_size, in_channels, T, dtype=torch.float)

    print("-"*10, "SiLU1d w/o padding", "-"*10)

    silu1d = SiLU1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    print(silu1d)
    output = silu1d(input)
    print(input.size(), output.size())
    print()

    print("-"*10, "SiLU1d w/ padding", "-"*10)
    padding = (stride - (T - kernel_size) % stride) % stride

    silu1d = SiLU1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    print(silu1d)
    output = silu1d(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Sigmoid Linear Units (SiLU) 1D", "="*10)
    _test_silu1d()
