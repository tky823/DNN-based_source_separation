import torch.nn as nn

def choose_nonlinear(name, **kwargs):
    if name == 'relu':
        nonlinear = nn.ReLU()
    elif name == 'sigmoid':
        nonlinear = nn.Sigmoid()
    elif name == 'leaky-relu':
        nonlinear = nn.LeakyReLU()
    else:
        raise NotImplementedError("Invalid nonlinear function is specified. Choose 'relu' instead of {}.".format(name))
    
    return nonlinear

def choose_rnn(name, **kwargs):
    if name == 'rnn':
        rnn = nn.RNN(**kwargs)
    elif name == 'lstm':
        rnn = nn.LSTM(**kwargs)
    elif name == 'gru':
        rnn = nn.GRU(**kwargs)
    else:
        raise NotImplementedError("Invalid nonlinear function is specified. Choose 'relu' instead of {}.".format(name))
    
    return rnn