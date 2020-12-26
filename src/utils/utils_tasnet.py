from models.tasnet import FourierEncoder, FourierDecoder, Encoder, Decoder
from norm import GlobalLayerNorm, CumulativeLayerNorm1d

EPS=1e-12

def choose_bases(hidden_channels, kernel_size, stride=None, enc_bases='trainable', dec_bases='trainable', **kwargs):
    if enc_bases == 'trainable':
        encoder = Encoder(1, hidden_channels, kernel_size, stride=stride, nonlinear=kwargs['enc_nonlinear'])
    elif enc_bases == 'Fourier':
        encoder = FourierEncoder(1, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif enc_bases == 'trainableFourier':
        encoder = FourierEncoder(1, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    else:
        raise NotImplementedError("Not support {} for encoder".format(enc_bases))
        
    if dec_bases == 'trainable':
        decoder = Decoder(hidden_channels, 1, kernel_size, stride=stride)
    elif dec_bases == 'Fourier':
        decoder = FourierDecoder(hidden_channels, 1, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif dec_bases == 'trainableFourier':
        decoder = FourierDecoder(hidden_channels, 1, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    else:
        raise NotImplementedError("Not support {} for decoder".format(dec_bases))
        
    return encoder, decoder

def choose_layer_norm(num_features, causal=False, eps=EPS):
    if causal:
        norm = CumulativeLayerNorm1d(num_features, eps=eps)
    else:
        norm = GlobalLayerNorm(num_features, eps=eps)
    
    return norm
