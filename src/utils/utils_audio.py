import math
from scipy.io import wavfile
import numpy as np
import torch

def read_wav(path):
    sr, signal = wavfile.read(path)
    signal = signal / 32768
    
    return signal, sr

def write_wav(path, signal, sr):
    signal = signal * 32768
    signal = np.clip(signal, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, signal)

def build_Fourier_basis(fft_size, normalize=False):
    """
    Args:
        fft_size <int>:
        normalize <bool>:
    """
    k = torch.arange(0, fft_size, dtype=torch.float)
    n = torch.arange(0, fft_size, dtype=torch.float)
    k, n = torch.meshgrid(k, n)

    cos_basis = torch.cos(2*math.pi*k*n/fft_size)
    sin_basis = torch.sin(2*math.pi*k*n/fft_size)
    
    if normalize:
        norm = math.sqrt(fft_size)
        cos_basis = cos_basis / norm
        sin_basis = sin_basis / norm
    
    return cos_basis, sin_basis
    
def build_window(fft_size, window_fn='hann'):
    if window_fn=='hann':
        window = torch.hann_window(fft_size, periodic=True)
    elif window_fn=='hamming':
        window = torch.hamming_window(fft_size, periodic=True)
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
        hop_size = window_length//2

    windows = torch.cat([
        torch.roll(window.unsqueeze(dim=0), hop_size*idx) for idx in range(window_length//hop_size)
    ], dim=0)
    
    power = windows**2
    norm = power.sum(dim=0)
    optimal_window = window / norm
    
    return optimal_window
