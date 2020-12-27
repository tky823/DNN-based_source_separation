import torch

EPS=1e-12

def ideal_binary_mask(input):
    """
    Args:
        input (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    Returns:
        mask (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    """
    n_dim = input.dim()
    
    if n_dim == 3:
        n_sources, F_bin, T_bin = input.size()
        
        input = input.permute(1,2,0).contiguous()
        flatten_input = input.view(F_bin*T_bin, n_sources)
        flatten_idx = torch.arange(0, F_bin*T_bin*n_sources, n_sources)
        flatten_idx = flatten_idx + flatten_input.argmax(dim=1)
        flatten_mask = torch.zeros(F_bin*T_bin*n_sources)
        flatten_mask[flatten_idx] = 1
        
        mask = flatten_mask.view(F_bin, T_bin, n_sources)
        mask = mask.permute(2,0,1).contiguous()
    elif n_dim == 4:
        batch_size, n_sources, F_bin, T_bin = input.size()
        
        input = input.permute(0,2,3,1).contiguous()
        flatten_input = input.view(batch_size*F_bin*T_bin, n_sources)
        flatten_idx = torch.arange(0, batch_size*F_bin*T_bin*n_sources, n_sources)
        flatten_idx = flatten_idx + flatten_input.argmax(dim=1)
        flatten_mask = torch.zeros(batch_size*F_bin*T_bin*n_sources)
        flatten_mask[flatten_idx] = 1
        
        mask = flatten_mask.view(batch_size, F_bin, T_bin, n_sources)
        mask = mask.permute(0,3,1,2).contiguous()
    else:
        raise ValueError("Not support {}-dimension".format(n_dim))
    
    return mask
    
def ideal_ratio_mask(input, eps=EPS):
    """
    Args:
        input (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    Returns:
        mask (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    """
    n_dim = input.dim()
    
    if n_dim == 3:
        norm = input.sum(dim=0, keepdim=True) # (1, F_bin, T_bin)
    elif n_dim == 4:
        norm = input.sum(dim=1, keepdim=True) # (batch_size, 1, F_bin, T_bin)
    else:
        raise ValueError("Not support {}-dimension".format(n_dim))
    
    mask = input / (norm + eps) # (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    
    return mask

def wiener_filter_mask(input, eps=EPS):
    """
    Args:
        input (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    Returns:
        mask (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    """
    n_dim = input.dim()
    power = input**2 # (n_sources, F_bin, T_bin) or (batch_size, n_sources, F_bin, T_bin)
    
    if n_dim == 3:
        norm = power.sum(dim=0, keepdim=True) # (1, F_bin, T_bin)
    elif n_dim == 4:
        norm = power.sum(dim=1, keepdim=True) # (batch_size, 1, F_bin, T_bin)
    else:
        raise ValueError("Not support {}-dimension".format(n_dim))
    
    mask = power / (norm + eps)

    return mask

"""
Phase sensitive mask
See "Phase-Sensitive and Recognition-Boosted Speech Separation using Deep Recurrent Neural Networks"
"""
def phase_sensitive_mask(input, eps=EPS):
    """
    Args:
        input (n_sources, 2*F_bin, T_bin) or (batch_size, 2*n_sources, F_bin, T_bin)
    Returns:
        mask (n_sources, 2*F_bin, T_bin) or (batch_size, 2*n_sources, F_bin, T_bin)
    """
    raise NotImplementedError("No implementation")

def _test(method='IBM'):
    if method == 'IBM':
        mask = ideal_binary_mask(amplitude)
    elif method == 'IRM':
        mask = ideal_ratio_mask(amplitude)
    elif method == 'WFM':
        mask = wiener_filter_mask(amplitude)
    else:
        raise NotImplementedError("Not support {}".format(method))
    
    estimated_amplitude = amplitude * mask
    
    real, imag = estimated_amplitude * torch.cos(phase_mixture), estimated_amplitude * torch.sin(phase_mixture)
    estimated_spectrgram = torch.cat([real, imag], dim=1)
    estimated_signal = istft(estimated_spectrgram, T=T)
    estimated_signal = estimated_signal.detach().cpu().numpy()
    
    for signal, tag in zip(estimated_signal, ['man', 'woman']):
        write_wav("data/frequency_mask/{}-estimated_{}.wav".format(tag, method), signal=signal, sr=16000)


if __name__ == '__main__':
    import os
    import numpy as np
    from scipy.signal import resample_poly
    
    from utils.utils_audio import read_wav, write_wav
    from stft import BatchSTFT, BatchInvSTFT
    
    os.makedirs("data/frequency_mask", exist_ok=True)
    
    fft_size, hop_size = 1024, 256
    n_basis = 4
    
    source1, sr = read_wav("data/man-44100.wav")
    source1 = resample_poly(source1, up=16000, down=sr)
    write_wav("data/man-16000.wav", signal=source1, sr=16000)
    T = len(source1)
    
    source2, sr = read_wav("data/woman-44100.wav")
    source2 = resample_poly(source2, up=16000, down=sr)
    write_wav("data/woman-16000.wav", signal=source2, sr=16000)
    
    mixture = source1 + source2
    write_wav("data/mixture-16000.wav", signal=mixture, sr=16000)
    
    stft = BatchSTFT(fft_size=fft_size, hop_size=hop_size)
    istft = BatchInvSTFT(fft_size=fft_size, hop_size=hop_size)
    
    mixture = torch.Tensor(mixture).unsqueeze(dim=0)
    source1 = torch.Tensor(source1).unsqueeze(dim=0)
    source2 = torch.Tensor(source2).unsqueeze(dim=0)
    
    spectrogram_mixture = stft(mixture)
    real = spectrogram_mixture[:,:fft_size//2+1]
    imag = spectrogram_mixture[:,fft_size//2+1:]
    power = real**2+imag**2
    amplitude_mixture = torch.sqrt(power)
    phase_mixture = torch.atan2(imag, real)
    
    spectrogram_source1 = stft(source1)
    real = spectrogram_source1[:,:fft_size//2+1]
    imag = spectrogram_source1[:,fft_size//2+1:]
    power = real**2+imag**2
    amplitude_source1 = torch.sqrt(power)
    
    spectrogram_source2 = stft(source2)
    real = spectrogram_source2[:,:fft_size//2+1]
    imag = spectrogram_source2[:,fft_size//2+1:]
    power = real**2+imag**2
    amplitude_source2 = torch.sqrt(power)

    amplitude = torch.cat([amplitude_source1, amplitude_source2], dim=0)

    _test('IBM')
    _test('IRM')
    _test('WFM')