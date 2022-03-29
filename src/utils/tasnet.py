import warnings

import torch.nn as nn

from utils.filterbank import choose_filterbank
from modules.norm import GlobalLayerNorm, CumulativeLayerNorm1d

EPS = 1e-12

def choose_basis(hidden_channels, kernel_size, stride=None, enc_basis='trainable', dec_basis='trainable', **kwargs):
    warnings.warn("Use utils.filterbank.choose_filterbank instead.", DeprecationWarning)
    return choose_filterbank(hidden_channels, kernel_size, stride=stride, enc_basis=enc_basis, dec_basis=dec_basis, **kwargs)

def choose_layer_norm(name, num_features, causal=False, eps=EPS, **kwargs):
    if name == 'cLN':
        layer_norm = CumulativeLayerNorm1d(num_features, eps=eps)
    elif name == 'gLN':
        if causal:
            raise ValueError("Global Layer Normalization is NOT causal.")
        layer_norm = GlobalLayerNorm(num_features, eps=eps)
    elif name in ['BN', 'batch', 'batch_norm']:
        n_dims = kwargs.get('n_dims') or 1
        if n_dims == 1:
            layer_norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            layer_norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError("n_dims is expected 1 or 2, but give {}.".format(n_dims))
    else:
        raise NotImplementedError("Not support {} layer normalization.".format(name))
    
    return layer_norm