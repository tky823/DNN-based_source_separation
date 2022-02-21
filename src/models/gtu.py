import torch
import torch.nn as nn

"""
Gated Tanh Units
Reference: "Language Modeling with Gated Convolutional Network"
See https://arxiv.org/abs/1612.08083
"""

class GTU1d(nn.Module):
    """
    Gated Tanh Units for 1D inputs
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
        self.map_gate = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, T)
        Returns:
            output (batch_size, out_channels, T)
        """
        x_output = self.map(input)
        x_output = torch.tanh(x_output)
        x_gate = self.map_gate(input)
        x_gate = torch.sigmoid(x_gate)

        output = x_output * x_gate

        return output

class GTU2d(nn.Module):
    """
    Gated Tanh Units for 2D inputs
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
        self.map_gate = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, H, W)
        Returns:
            output (batch_size, out_channels, H, W)
        """
        x_output = self.map(input)
        x_output = torch.tanh(x_output)
        x_gate = self.map_gate(input)
        x_gate = torch.sigmoid(x_gate)

        output = x_output * x_gate

        return output

def _test_gtu1d():
    torch.manual_seed(111)

    batch_size = 4
    in_channels, out_channels = 3, 16
    kernel_size, stride = 3, 2
    T = 128

    input = torch.rand(batch_size, in_channels, T, dtype=torch.float)

    print("-"*10, "GTU1d w/o padding", "-"*10)

    glu1d = GTU1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    print(glu1d)
    output = glu1d(input)
    print(input.size(), output.size())
    print()

    print("-"*10, "GTU1d w/ padding", "-"*10)
    padding = (stride - (T - kernel_size) % stride) % stride

    gtu1d = GTU1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    print(gtu1d)
    output = gtu1d(input)
    print(input.size(), output.size())

def _test_gtu2d():
    torch.manual_seed(111)

    batch_size = 4
    in_channels, out_channels = 3, 16
    kernel_size, stride = 3, 2
    H, W = 512, 256

    input = torch.rand(batch_size, in_channels, H, W, dtype=torch.float)

    print("-"*10, "GTU2d", "-"*10)

    glu2d = GTU2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    print(glu2d)
    output = glu2d(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Gated Tanh Units (GTU) 1D", "="*10)
    _test_gtu1d()
    print()

    print("="*10, "Gated Tanh Units (GTU) 2D", "="*10)
    _test_gtu2d()
