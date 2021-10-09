from models.filterbank import FourierEncoder, FourierDecoder, Encoder, Decoder, GatedEncoder, PinvDecoder

EPS = 1e-12

def choose_filterbank(hidden_channels, kernel_size, stride=None, enc_basis='trainable', dec_basis='trainable', **kwargs):
    in_channels = kwargs.get('in_channels') or 1
    
    if enc_basis == 'trainable':
        if dec_basis == 'pinv':
            encoder = Encoder(in_channels, hidden_channels, kernel_size, stride=stride)
        else:
            encoder = Encoder(in_channels, hidden_channels, kernel_size, stride=stride, nonlinear=kwargs['enc_nonlinear'])
    elif enc_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']:
        assert_monoral(in_channels)
        trainable = False if enc_basis == 'Fourier' else True
        trainable_phase = True if enc_basis == 'trainableFourierTrainablePhase' else False
        onesided, return_complex = bool(kwargs['enc_onesided']), bool(kwargs['enc_return_complex'])
        window_fn = kwargs['window_fn']
        n_basis = compute_valid_basis(hidden_channels, onesided=onesided, return_complex=return_complex)
        encoder = FourierEncoder(n_basis, kernel_size, stride=stride, window_fn=window_fn, trainable=trainable, trainable_phase=trainable_phase, onesided=onesided, return_complex=return_complex)
    elif enc_basis == 'trainableGated':
        eps = kwargs.get('eps') or EPS
        encoder = GatedEncoder(in_channels, hidden_channels, kernel_size=kernel_size, stride=stride, eps=eps)
    else:
        raise NotImplementedError("Not support {} for encoder".format(enc_basis))
        
    if dec_basis == 'trainable':
        decoder = Decoder(hidden_channels, in_channels, kernel_size, stride=stride)
    elif dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']:
        assert_monoral(in_channels)
        trainable = False if dec_basis == 'Fourier' else True
        trainable_phase = True if dec_basis == 'trainableFourierTrainablePhase' else False
        onesided, return_complex = bool(kwargs['enc_onesided']), bool(kwargs['enc_return_complex'])
        window_fn = kwargs['window_fn']
        n_basis = compute_valid_basis(hidden_channels, onesided=onesided, return_complex=return_complex)
        decoder = FourierDecoder(n_basis, kernel_size, stride=stride, window_fn=window_fn, trainable=trainable, trainable_phase=trainable_phase, onesided=onesided)
    elif dec_basis == 'pinv':
        if enc_basis in ['trainable', 'trainableFourier', 'trainableFourierTrainablePhase']:
            assert_monoral(in_channels)
            decoder = PinvDecoder(encoder)
        else:
            raise NotImplementedError("Not support {} for decoder".format(dec_basis))
    else:
        raise NotImplementedError("Not support {} for decoder".format(dec_basis))
        
    return encoder, decoder

def assert_monoral(in_channels):
    # TODO: stereo input
    assert in_channels == 1, "`in_channels` is expected 1, but given {}.".format(in_channels)

def compute_valid_basis(hidden_channels, onesided=True, return_complex=True):
    if onesided:
        if return_complex:
            assert hidden_channels % 2 == 1, "`hidden_channels` is expected odd."
            n_basis = 2 * (hidden_channels - 1)
        else:
            assert hidden_channels % 2 == 0, "`hidden_channels` is expected even."
            n_basis = 2 * (hidden_channels // 2 - 1)
    else:
        if return_complex:
            n_basis = hidden_channels
        else:
            assert hidden_channels % 2 == 0, "`hidden_channels` is expected even."
            n_basis = hidden_channels // 2
    
    return n_basis