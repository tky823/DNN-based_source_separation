import math
import warnings

import numpy as np
import torch

import utils.audio as backend

def read_wav(path):
    from scipy.io import wavfile
    warnings.warn("Use torchaudio.load instead.", DeprecationWarning)

    sample_rate, signal = wavfile.read(path)
    signal = signal / 32768
    
    return signal, sample_rate

def write_wav(path, signal, sample_rate):
    from scipy.io import wavfile
    warnings.warn("Use torchaudio.save instead.", DeprecationWarning)

    signal = signal * 32768
    signal = np.clip(signal, -32768, 32767).astype(np.int16)
    wavfile.write(path, sample_rate, signal)

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
    warnings.warn("Use utils.audio.build_window instead.", DeprecationWarning)
    
    return backend.build_window(fft_size, window_fn=window_fn, **kwargs)
    
def build_optimal_window(window, hop_size=None):
    """
    Args:
        window: (window_length,)
    """
    warnings.warn("Use utils.audio.build_optimal_window instead.", DeprecationWarning)
    
    return backend.build_optimal_window(window, hop_size=hop_size)
