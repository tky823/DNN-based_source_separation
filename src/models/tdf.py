import torch
import torch.nn as nn

class TDF2d(nn.Module):
    """
    Time-Distributed Fully-connected Layer for time-frequency representation
    """
    def __init__(self, num_features, in_bins, out_bins, bias=False, nonlinear='relu'):
        super().__init__()

        self.net = TransformBlock2d(num_features, in_bins, out_bins, bias=bias, nonlinear=nonlinear)

    def forward(self, input):
        output = self.net(input)

        return output

class MultiheadTDF2d(nn.Module):
    def __init__(self, num_features, in_bins, out_bins, num_heads, bias=False, nonlinear='relu', stack_dim=1):
        super().__init__()

        self.num_heads = num_heads
        self.stack_dim = stack_dim

        net = []

        for idx in range(num_heads):
            net.append(TransformBlock2d(num_features, in_bins, out_bins, bias=bias, nonlinear=nonlinear))

        self.net = nn.ModuleList(net)

    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, in_channels, n_bins, n_frames) if stack_dim=1
        Returns:
            output <torch.Tensor>: (batch_size, num_heads, in_channels, n_bins, n_frames) if stack_dim=1
        """
        output = []

        for idx in range(self.num_heads):
            x = self.net[idx](input)
            output.append(x)
        
        output = torch.stack(output, dim=self.stack_dim)

        return output

class TransformBlock2d(nn.Module):
    def __init__(self, num_features, in_bins, out_bins, bias=False, nonlinear='relu'):
        super().__init__()

        self.nonlinear = nonlinear

        self.conv1d = nn.Conv1d(in_bins, out_bins, kernel_size=1, stride=1, bias=bias)
        self.norm2d = nn.BatchNorm2d(num_features)
        if nonlinear == 'relu':
            self.nonlinear2d = nn.ReLU()
    
    def forward(self, input):
        """
        Args:
            input <torch.Tensor>: (batch_size, num_features, in_bins, n_frames)
        Returns:
            output <torch.Tensor>: (batch_size, num_features, out_bins, n_frames)
        """
        batch_size, num_features, _, n_frames = input.size()

        x = input.view(batch_size * num_features, -1, n_frames)
        x = self.conv1d(x)
        x = x.view(batch_size, num_features, -1, n_frames)
        x = self.norm2d(x)

        if self.nonlinear:
            output = self.nonlinear2d(x)
        else:
            output = x
        
        return output
