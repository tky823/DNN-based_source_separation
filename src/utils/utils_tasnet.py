from models.tasnet import FourierEncoder, FourierDecoder, Encoder, Decoder
from norm import GlobalLayerNorm, CumulativeLayerNorm1d

EPS=1e-12

def choose_basis(hidden_channels, kernel_size, stride=None, enc_basis='trainable', dec_basis='trainable', **kwargs):
    if enc_basis == 'trainable':
        encoder = Encoder(1, hidden_channels, kernel_size, stride=stride, nonlinear=kwargs['enc_nonlinear'])
    elif enc_basis == 'Fourier':
        encoder = FourierEncoder(1, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif enc_basis == 'trainableFourier':
        encoder = FourierEncoder(1, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    else:
        raise NotImplementedError("Not support {} for encoder".format(enc_basis))
        
    if dec_basis == 'trainable':
        decoder = Decoder(hidden_channels, 1, kernel_size, stride=stride)
    elif dec_basis == 'Fourier':
        decoder = FourierDecoder(hidden_channels, 1, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif dec_basis == 'trainableFourier':
        decoder = FourierDecoder(hidden_channels, 1, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    else:
        raise NotImplementedError("Not support {} for decoder".format(dec_basis))
        
    return encoder, decoder

def choose_layer_norm(num_features, causal=False, eps=EPS):
    if causal:
        norm = CumulativeLayerNorm1d(num_features, eps=eps)
    else:
        norm = GlobalLayerNorm(num_features, eps=eps)
    
    return norm
