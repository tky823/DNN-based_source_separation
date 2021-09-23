import torch.nn as nn

from models.filterbank import FourierEncoder, FourierDecoder, Encoder, Decoder, PinvEncoder
from norm import GlobalLayerNorm, CumulativeLayerNorm1d

EPS = 1e-12

def choose_basis(hidden_channels, kernel_size, stride=None, enc_basis='trainable', dec_basis='trainable', **kwargs):
    in_channels = kwargs.get('in_channels') or 1
    
    if enc_basis == 'trainable':
        if dec_basis == 'pinv':
            encoder = Encoder(in_channels, hidden_channels, kernel_size, stride=stride)
        else:
            encoder = Encoder(in_channels, hidden_channels, kernel_size, stride=stride, nonlinear=kwargs['enc_nonlinear'])
    elif enc_basis in ['Fourier', 'trainableFourier']:
        assert_monoral(in_channels)
        trainable = False if 'Fourier' else True
        onesided, return_complex = kwargs['onesided'], kwargs['return_complex']
        window_fn = kwargs['window_fn']
        assert_hidden_channels(hidden_channels, kernel_size, onesided=onesided, return_complex=return_complex)
        encoder = FourierEncoder(kernel_size, stride=stride, window_fn=window_fn, trainable=trainable, onesided=onesided, return_complex=return_complex)
    else:
        raise NotImplementedError("Not support {} for encoder".format(enc_basis))
        
    if dec_basis == 'trainable':
        decoder = Decoder(hidden_channels, in_channels, kernel_size, stride=stride)
    elif dec_basis in ['Fourier', 'trainableFourier']:
        assert_monoral(in_channels)
        trainable = False if 'Fourier' else True
        onesided, return_complex = kwargs['onesided'], kwargs['return_complex']
        window_fn = kwargs['window_fn']
        assert_hidden_channels(hidden_channels, kernel_size, onesided=onesided, return_complex=return_complex)
        decoder = FourierDecoder(kernel_size, stride=stride, window_fn=window_fn, trainable=trainable, onesided=onesided)
    elif dec_basis == 'pinv':
        if enc_basis in ['trainable', 'trainableFourier']:
            assert_monoral(in_channels)
            decoder = PinvEncoder(encoder)
        else:
            raise NotImplementedError("Not support {} for decoder".format(dec_basis))
    else:
        raise NotImplementedError("Not support {} for decoder".format(dec_basis))
        
    return encoder, decoder

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

def assert_monoral(in_channels):
    # TODO: stereo input
    assert in_channels == 1, "`in_channels` is expected 1, but given {}.".format(in_channels)

def assert_hidden_channels(hidden_channels, kernel_size, onesided=True, return_complex=True):
    if onesided:
        if return_complex:
            assert hidden_channels == kernel_size // 2 + 1, "`hidden_channels` is expected equal to `kernel_size // 2 + 1`."
        else:
            assert hidden_channels == 2 * (kernel_size // 2 + 1), "`hidden_channels` is expected equal to `2 * (kernel_size // 2 + 1)`."
    else:
        if return_complex:
            assert hidden_channels == kernel_size, "`hidden_channels` is expected equal to `kernel_size`."
        else:
            assert hidden_channels == 2 * kernel_size, "`hidden_channels` is expected equal to `2 * kernel_size`."