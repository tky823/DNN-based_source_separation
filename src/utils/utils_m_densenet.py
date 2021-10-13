import torch.nn as nn

EPS = 1e-12

def choose_layer_norm(name, num_features, n_dims=2, eps=EPS, **kwargs):
    if name in ['BN', 'batch', 'batch_norm']:
        if n_dims == 1:
            layer_norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            layer_norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError("n_dims is expected 1 or 2, but give {}.".format(n_dims))
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))
    
    return layer_norm

def choose_nonlinear(name, **kwargs):
    if name == 'relu':
        nonlinear = nn.ReLU()
    else:
        raise NotImplementedError("Invalid nonlinear function is specified. Choose 'relu' instead of {}.".format(name))
    
    return nonlinear