import torch.nn as nn

from models.dense_lstm import DenseRNNParallelBlock, RNNBeforeDenseBlock, RNNAfterDenseBlock

EPS = 1e-12

def choose_dense_rnn_block(rnn_type, rnn_position, in_channels, growth_rate, kernel_size, dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
    if rnn_position == 'after_dense':
        block = RNNAfterDenseBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, rnn_type=rnn_type, eps=eps)
    elif rnn_position == 'before_dense':
        block = RNNBeforeDenseBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, rnn_type=rnn_type, eps=eps)
    elif rnn_position == 'parallel':
        block = DenseRNNParallelBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, rnn_type=rnn_type, eps=eps)
    else:
        raise NotImplementedError("Invalid RNN position is specified. Choose 'after_dense', 'before_dense', or 'parallel' instead of {}.".format(rnn_position))
    
    return block