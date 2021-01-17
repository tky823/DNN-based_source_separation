from models.tasnet import FourierEncoder, FourierDecoder, Encoder, Decoder, PinvEncoder
from models.eliya_net import OverlapAddDecoder

EPS=1e-12

def choose_bases(hidden_channels, kernel_size, stride=None, enc_bases='trainable', dec_bases='trainable', **kwargs):
    if enc_bases == 'trainable':
        if dec_bases == 'pinv':
            encoder = Encoder(1, hidden_channels, kernel_size, stride=stride)
        else:
            encoder = Encoder(1, hidden_channels, kernel_size, stride=stride, nonlinear=kwargs['enc_nonlinear'])
    elif enc_bases == 'Fourier':
        encoder = FourierEncoder(1, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif enc_bases == 'trainableFourier':
        encoder = FourierEncoder(1, hidden_channels, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    else:
        raise NotImplementedError("Not support {} for encoder".format(enc_bases))
    
    if dec_bases == 'overlap-add':
        decoder = OverlapAddDecoder(hidden_channels, 1, kernel_size, stride=stride)
    elif dec_bases == 'trainable':
        decoder = Decoder(hidden_channels, 1, kernel_size, stride=stride)
    elif dec_bases == 'Fourier':
        decoder = FourierDecoder(hidden_channels, 1, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=False)
    elif dec_bases == 'trainableFourier':
        decoder = FourierDecoder(hidden_channels, 1, kernel_size, stride=stride, window_fn=kwargs['window_fn'], trainable=True)
    elif dec_bases == 'pinv':
        if enc_bases == 'trainable':
            decoder = PinvEncoder(encoder)
        else:
            raise NotImplementedError("Not support {} for decoder".format(dec_bases))
    else:
        raise NotImplementedError("Not support {} for decoder".format(dec_bases))
        
    return encoder, decoder