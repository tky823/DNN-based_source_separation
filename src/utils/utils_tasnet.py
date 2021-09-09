import torch.nn as nn

from models.tasnet import FourierEncoder, FourierDecoder, Encoder, Decoder, PinvEncoder
from norm import GlobalLayerNorm, CumulativeLayerNorm1d

EPS=1e-12

def choose_bases(hidden_channels, kernel_size, stride=None, enc_bases='trainable', dec_bases='trainable', **kwargs):
    if 'in_channels' in kwargs:
        in_channels = kwargs['in_channels']
    else:
        in_channels = 1
    
    if enc_bases == 'trainable':
        if dec_bases == 'pinv':
            encoder = Encoder(in_channels, hidden_channels, kernel_size, stride=stride)
        else:
            encoder = Encoder(in_channels, hidden_channels, kernel_size, stride=stride, nonlinear=kwargs['enc_nonlinear'])
    elif enc_bases == 'Fourier':
        assert in_channels == 1 # TODO
        encoder = FourierEncoder(in_channels, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif enc_bases == 'trainableFourier':
        assert in_channels == 1 # TODO
        encoder = FourierEncoder(in_channels, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    else:
        raise NotImplementedError("Not support {} for encoder".format(enc_bases))
        
    if dec_bases == 'trainable':
        decoder = Decoder(hidden_channels, in_channels, kernel_size, stride=stride)
    elif dec_bases == 'Fourier':
        assert in_channels == 1 # TODO
        decoder = FourierDecoder(hidden_channels, in_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif dec_bases == 'trainableFourier':
        assert in_channels == 1 # TODO
        decoder = FourierDecoder(hidden_channels, in_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    elif dec_bases == 'pinv':
        if enc_bases == 'trainable':
            assert in_channels == 1 # TODO
            decoder = PinvEncoder(encoder)
        else:
            raise NotImplementedError("Not support {} for decoder".format(dec_bases))
    else:
        raise NotImplementedError("Not support {} for decoder".format(dec_bases))
        
    return encoder, decoder

def choose_layer_norm(name, num_features, causal=False, eps=EPS, **kwargs):
    if name == 'cLM':
        layer_norm = CumulativeLayerNorm1d(num_features, eps=eps)
    elif name == 'gLM':
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
