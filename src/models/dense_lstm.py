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
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, dilated=False, norm=True, nonlinear='relu', rnn_type='parallel', eps=EPS):
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

        self.rnn = choose_rnn(rnn_type)
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
        x = self.dense_block(input)
        output, _ = self.rnn(x)

        return output

class RNNBeforeDenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, dilated=False, norm=True, nonlinear='relu', rnn_type='parallel', eps=EPS):
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

        self.rnn = choose_rnn(rnn_type)
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
        x, _ = self.rnn(input)
        output = self.dense_block(x)

        return output

class DenseRNNParallelBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size, depth=None, dilated=False, norm=True, nonlinear='relu', rnn_type='parallel', eps=EPS):
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

        self.dense_block = DenseBlock(
            in_channels, growth_rate, kernel_size,
            depth=depth, dilated=dilated, norm=norm, nonlinear=nonlinear,
            eps=eps
        )
        self.rnn = choose_rnn(rnn_type)
    
    def forward(self, input):
        """
        Args:
            input: (batch_size, in_channels, H, W)
        Returns:
            output: (batch_size, out_channels, H, W), where `out_channels` is determined by `growth_rate`.
        """
        x_dense = self.dense_block(input)
        x_rnn, _ = self.rnn(input)
        output = x_dense + x_rnn

        return output
