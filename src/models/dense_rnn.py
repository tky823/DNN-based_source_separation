import torch.nn as nn

from utils.utils_model import choose_rnn
from models.m_densenet import DenseBlock

"""
Reference: MMDenseLSTM: An efficient combination of convolutional and recurrent neural networks for audio source separation
See https://ieeexplore.ieee.org/document/8521383
"""

FULL = 'full'
EPS = 1e-12

class RNNAfterDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, n_bins=None, depth=None, dilated=False, norm=True, nonlinear='relu', causal=False, rnn_type='rnn', hidden_channels=None, eps=EPS, **rnn_kwargs):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.dense_block = DenseBlock(
            in_channels, growth_rate, kernel_size,
            depth=depth, dilated=dilated, norm=norm, nonlinear=nonlinear,
            eps=eps
        )
        out_channels = self.dense_block.out_channels
        self.bottleneck_conv2d = nn.Conv2d(out_channels, 1, kernel_size=(1, 1))
        self.rnn = choose_rnn(rnn_type, input_size=n_bins, hidden_size=hidden_channels, batch_first=True, bidirectional=bidirectional, **rnn_kwargs)
        self.linear = nn.Linear(num_directions * hidden_channels, out_channels * n_bins)
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by `growth_rate`.
        """
        batch_size, _, H, W = input.size()
        out_channels = self.dense_block.out_channels

        x = self.dense_block(input)
        x = self.bottleneck_conv2d(x)
        x = x.squeeze(dim=1)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = x.view(batch_size, W, out_channels, H)
        output = x.permute(0, 2, 3, 1).contiguous()

        return output

class RNNBeforeDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, n_bins=None, depth=None, dilated=False, norm=True, nonlinear='relu', causal=False, rnn_type='rnn', hidden_channels=None, eps=EPS, **rnn_kwargs):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.bottleneck_conv2d = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.rnn = choose_rnn(rnn_type, input_size=n_bins, hidden_size=hidden_channels, batch_first=True, bidirectional=bidirectional, **rnn_kwargs)
        self.linear = nn.Linear(num_directions * hidden_channels, in_channels * n_bins)
        self.dense_block = DenseBlock(
            in_channels, growth_rate, kernel_size,
            depth=depth, dilated=dilated, norm=norm, nonlinear=nonlinear,
            eps=eps
        )
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by `growth_rate`.
        """
        batch_size, in_channels, H, W = input.size()

        x = self.bottleneck_conv2d(input)
        x = x.squeeze(dim=1)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.rnn(x)
        x = self.linear(x)
        x = x.view(batch_size, W, in_channels, H)
        x = x.permute(0, 2, 3, 1).contiguous()
        output = self.dense_block(x)

        return output

class DenseRNNParallelBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, n_bins=None, depth=None, dilated=False, norm=True, nonlinear='relu', causal=False, rnn_type='rnn', hidden_channels=None, eps=EPS, **rnn_kwargs):
        """
        Args:
            in_channels <int>: # of input channels
            growth_rate <int> or <list<int>>: # of output channels
            kernel_size <int> or <tuple<int>>: Kernel size
            dilated <bool> or <list<bool>>: Applies dilated convolution.
            norm <bool> or <list<bool>>: Applies batch normalization.
            nonlinear <str> or <list<str>>: Applies nonlinear function.
            depth <int>: If `growth_rate` is given by list, len(growth_rate) must be equal to `depth`.
        """
        super().__init__()

        if causal:
            bidirectional = False
            num_directions = 1
        else:
            bidirectional = True
            num_directions = 2

        self.dense_block = DenseBlock(
            in_channels, growth_rate, kernel_size,
            depth=depth, dilated=dilated, norm=norm, nonlinear=nonlinear,
            eps=eps
        )
        out_channels = self.dense_block.out_channels
        self.bottleneck_conv2d = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.rnn = choose_rnn(rnn_type, input_size=n_bins, hidden_size=hidden_channels, batch_first=True, bidirectional=bidirectional, **rnn_kwargs)
        self.linear = nn.Linear(num_directions * hidden_channels, out_channels * n_bins)
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by `growth_rate`.
        """
        batch_size, _, H, W = input.size()
        out_channels = self.dense_block.out_channels

        x_dense = self.dense_block(input)
        x_rnn = self.bottleneck_conv2d(input)
        x_rnn = x_rnn.squeeze(dim=1)
        x_rnn = x_rnn.permute(0, 2, 1).contiguous()
        x_rnn, _ = self.rnn(x_rnn)
        x_rnn = self.linear(x_rnn)
        x_rnn = x_rnn.view(batch_size, W, out_channels, H)
        x_rnn = x_rnn.permute(0, 2, 3, 1).contiguous()
        output = x_dense + x_rnn

        return output

def _test_rnn_after_dense():
    batch_size = 4
    C = 8
    H, W = 17, 10
    K = 3
    growth_rate = 2

    input = torch.randn((batch_size, C, H, W), dtype=torch.float)

    model = RNNAfterDenseBlock(C, growth_rate, kernel_size=K, n_bins=H, depth=7, hidden_channels=6)

    print(model)

    output = model(input)

    print(input.size(), output.size())

def _test_rnn_before_dense():
    batch_size = 4
    C = 8
    H, W = 17, 10
    K = 3
    growth_rate = 2

    input = torch.randn((batch_size, C, H, W), dtype=torch.float)

    model = RNNBeforeDenseBlock(C, growth_rate, kernel_size=K, n_bins=H, depth=7, hidden_channels=6)

    print(model)

    output = model(input)

    print(input.size(), output.size())

def _test_dense_rnn_parallel():
    batch_size = 4
    C = 8
    H, W = 17, 10
    K = 3
    growth_rate = 2

    input = torch.randn((batch_size, C, H, W), dtype=torch.float)

    model = DenseRNNParallelBlock(C, growth_rate, kernel_size=K, n_bins=H, depth=7, hidden_channels=6)

    print(model)

    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    print("="*10, "RNNAfterDenseBlock", "="*10)
    _test_rnn_after_dense()
    print()

    print("="*10, "RNNBeforeDenseBlock", "="*10)
    _test_rnn_before_dense()

    print("="*10, "DenseRNNParallelBlock", "="*10)
    _test_dense_rnn_parallel()