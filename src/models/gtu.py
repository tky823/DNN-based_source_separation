import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Gated Tanh Units
See "Language Modeling with Gated Convolutional Network"
https://arxiv.org/abs/1612.08083
"""

class GTU(nn.Module):
    """
    Gated Tanh Units
    You can use GTU1d and GTU2d as well.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels <int>
            out_channels <int>
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Linear(in_channels, out_channels)
        self.map_gate = nn.Linear(in_channels, out_channels)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, in_channels, *)
        Returns:
            output (batch_size, out_channels, *)
        """
        dim = input.dim()
        axis = tuple(range(dim))
        permutation = axis[0:1] + axis[:0:-1]
        
        input = input.permute(*permutation)
        x_output = self.map(input)
        x_output = torch.tanh(x_output)
        x_gate = self.map_gate(input)
        x_gate = torch.sigmoid(x_gate)

        output = x_output * x_gate
        output = output.permute(*permutation)
        
        return output


class GTU1d(nn.Module):
    """
    Gated Tanh Units for 1D inputs
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels <int>
            out_channels <int>
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.map_gate = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        
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
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels <int>
            out_channels <int>
        """
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
            
        self.in_channels, self.out_channels = in_channels, out_channels

        self.map = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        self.map_gate = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,1))
        
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
    T = 128

    input = torch.rand(batch_size, in_channels, T, dtype=torch.float)

    print("-"*10, "By using GLU", "-"*10)

    glu1d = GTU(in_channels, out_channels)
    print(glu1d)
    output = glu1d(input)
    print(input.size(), output.size())
    print()

    # 1-D
    print("-"*10, "By using GLU1d", "-"*10)

    glu1d = GTU1d(in_channels, out_channels)
    print(glu1d)
    output = glu1d(input)
    print(input.size(), output.size())

def _test_gtu2d():
    torch.manual_seed(111)

    batch_size = 4
    in_channels, out_channels = 3, 16
    H, W = 512, 256

    input = torch.rand(batch_size, in_channels, H, W, dtype=torch.float)

    print("-"*10, "By using GLU", "-"*10)
    
    glu2d = GTU(in_channels, out_channels)
    print(glu2d)
    output = glu2d(input)
    print(input.size(), output.size())
    print()

    print("-"*10, "By using GLU2d", "-"*10)

    glu2d = GTU2d(in_channels, out_channels)
    print(glu2d)
    output = glu2d(input)
    print(input.size(), output.size())

if __name__ == '__main__':
    print("="*10, "Gated Tanh Units (GTU) 1D", "="*10)
    _test_gtu1d()
    print()

    print("="*10, "Gated Tanh Units (GTU) 2D", "="*10)
    _test_gtu2d()
