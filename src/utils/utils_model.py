import torch.nn as nn

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