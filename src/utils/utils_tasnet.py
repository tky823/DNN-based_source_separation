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

def choose_layer_norm(num_features, causal=False, eps=EPS):
    if causal:
        norm = CumulativeLayerNorm1d(num_features, eps=eps)
    else:
        norm = GlobalLayerNorm(num_features, eps=eps)
    
    return norm
