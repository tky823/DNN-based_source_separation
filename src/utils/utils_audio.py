import math
import warnings

import numpy as np
import torch

def read_wav(path):
    from scipy.io import wavfile
    warnings.warn("Use torchaudio.load instead.", DeprecationWarning)

    sr, signal = wavfile.read(path)
    signal = signal / 32768
    
    return signal, sr

def write_wav(path, signal, sr):
    from scipy.io import wavfile
    warnings.warn("Use torchaudio.save instead.", DeprecationWarning)

    signal = signal * 32768
    signal = np.clip(signal, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, signal)

def mu_law_compand(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

def inv_mu_law_compand(y, mu=255):
    return np.sign(y) * ((1 + mu)**np.abs(y) - 1) / mu

def build_Fourier_bases(fft_size, normalize=False):
    """
    Args:
        fft_size <int>:
        normalize <bool>:
    """
    k = torch.arange(0, fft_size, dtype=torch.float)
    n = torch.arange(0, fft_size, dtype=torch.float)
    k, n = torch.meshgrid(k, n)

    cos_bases = torch.cos(2*math.pi*k*n/fft_size)
    sin_bases = torch.sin(2*math.pi*k*n/fft_size)
    
    if normalize:
        norm = math.sqrt(fft_size)
        cos_bases = cos_bases / norm
        sin_bases = sin_bases / norm
    
    return cos_bases, sin_bases
    
def build_window(fft_size, window_fn='hann', **kwargs):
    if window_fn=='hann':
        assert set(kwargs) == set(), "kwargs is expected empty but given kwargs={}.".format(kwargs)
        window = torch.hann_window(fft_size, periodic=True)
    elif window_fn=='hamming':
        assert set(kwargs) == set(), "kwargs is expected empty but given kwargs={}.".format(kwargs)
        window = torch.hamming_window(fft_size, periodic=True)
    elif window_fn == 'blackman':
        window = torch.blackman_window(fft_size, periodic=True)
    elif window_fn=='kaiser':
        assert set(kwargs) == {'beta'}, "kwargs is expected to include key `beta` but given kwargs={}.".format(kwargs)
        window = torch.kaiser_window(fft_size, beta=kwargs['beta'], periodic=True)
    else:
        raise ValueError("Not support {} window.".format(window_fn))
    
    return window
    
def build_optimal_window(window, hop_size=None):
    """
    Args:
        window: (window_length,)
    """
    window_length = len(window)

    if hop_size is None:
        hop_size = window_length // 2

    windows = torch.cat([
        torch.roll(window.unsqueeze(dim=0), hop_size*idx) for idx in range(window_length // hop_size)
    ], dim=0)
    
    power = windows**2
    norm = power.sum(dim=0)
    optimal_window = window / norm
    
    return optimal_window
