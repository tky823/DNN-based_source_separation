from models.dense_rnn import DenseRNNParallelBlock, RNNBeforeDenseBlock, RNNAfterDenseBlock

EPS = 1e-12

def choose_dense_rnn_block(rnn_type, rnn_position, in_channels, growth_rate, hidden_channels, kernel_size, n_bins=None, dilated=False, norm=True, nonlinear='relu', causal=False, depth=None, eps=EPS, **rnn_kwargs):
    if rnn_position == 'after_dense':
        block = RNNAfterDenseBlock(in_channels, growth_rate, kernel_size, n_bins=n_bins, dilated=dilated, norm=norm, nonlinear=nonlinear, causal=causal, depth=depth, rnn_type=rnn_type, hidden_channels=hidden_channels, eps=eps, **rnn_kwargs)
    elif rnn_position == 'before_dense':
        block = RNNBeforeDenseBlock(in_channels, growth_rate, kernel_size, n_bins=n_bins, dilated=dilated, norm=norm, nonlinear=nonlinear, causal=causal, depth=depth, rnn_type=rnn_type, hidden_channels=hidden_channels, eps=eps, **rnn_kwargs)
    elif rnn_position == 'parallel':
        block = DenseRNNParallelBlock(in_channels, growth_rate, kernel_size, n_bins=n_bins, dilated=dilated, norm=norm, nonlinear=nonlinear, causal=causal, depth=depth, rnn_type=rnn_type, hidden_channels=hidden_channels, eps=eps, **rnn_kwargs)
    else:
        raise NotImplementedError("Invalid RNN position is specified. Choose 'after_dense', 'before_dense', or 'parallel' instead of {}.".format(rnn_position))
    
    return block