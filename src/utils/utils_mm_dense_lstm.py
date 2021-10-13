import torch.nn as nn

from models.mm_dense_lstm import DenseRNNParallelBlock, RNNBeforeDenseBlock, RNNAfterDenseBlock

EPS = 1e-12

def choose_rnn(name, **kwargs):
    if name == 'rnn':
        rnn = nn.RNN(**kwargs)
    elif name == 'lstm':
        rnn = nn.LSTM(**kwargs)
    elif name == 'gru':
        rnn = nn.GRU(**kwargs)
    else:
        raise NotImplementedError("Invalid RNN is specified. Choose 'rnn', 'lstm', or 'gru' instead of {}.".format(name))
    
    return rnn

def choose_dense_rnn_block(rnn_position, in_channels, growth_rate, kernel_size, dilated=False, norm=True, nonlinear='relu', depth=None, eps=EPS):
    if rnn_position == 'after_dense':
        block = RNNAfterDenseBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
    elif rnn_position == 'before_dense':
        block = RNNBeforeDenseBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
    elif rnn_position == 'parallel':
        block = DenseRNNParallelBlock(in_channels, growth_rate, kernel_size, dilated=dilated, norm=norm, nonlinear=nonlinear, depth=depth, eps=eps)
    else:
        raise NotImplementedError("Invalid RNN position is specified. Choose 'after_dense', 'before_dense', or 'parallel' instead of {}.".format(rnn_position))
    
    return block