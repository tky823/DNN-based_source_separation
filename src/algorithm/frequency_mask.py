import torch

EPS=1e-12

def ideal_binary_mask(input):
    """
    Args:
        input (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    n_dim = input.dim()
    
    if n_dim == 3:
        n_sources, n_bins, n_frames = input.size()
        
        input = input.permute(1,2,0).contiguous()
        flatten_input = input.view(n_bins*n_frames, n_sources)
        flatten_idx = torch.arange(0, n_bins*n_frames*n_sources, n_sources)
        flatten_idx = flatten_idx + flatten_input.argmax(dim=1)
        flatten_mask = torch.zeros(n_bins*n_frames*n_sources)
        flatten_mask[flatten_idx] = 1
        
        mask = flatten_mask.view(n_bins, n_frames, n_sources)
        mask = mask.permute(2,0,1).contiguous()
    elif n_dim == 4:
        batch_size, n_sources, n_bins, n_frames = input.size()
        
        input = input.permute(0,2,3,1).contiguous()
        flatten_input = input.view(batch_size*n_bins*n_frames, n_sources)
        flatten_idx = torch.arange(0, batch_size*n_bins*n_frames*n_sources, n_sources)
        flatten_idx = flatten_idx + flatten_input.argmax(dim=1)
        flatten_mask = torch.zeros(batch_size*n_bins*n_frames*n_sources)
        flatten_mask[flatten_idx] = 1
        
        mask = flatten_mask.view(batch_size, n_bins, n_frames, n_sources)
        mask = mask.permute(0,3,1,2).contiguous()
    else:
        raise ValueError("Not support {}-dimension".format(n_dim))
    
    return mask
    
def ideal_ratio_mask(input, eps=EPS):
    """
    Args:
        input (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    n_dim = input.dim()
    
    if n_dim == 3:
        norm = input.sum(dim=0, keepdim=True) # (1, n_bins, n_frames)
    elif n_dim == 4:
        norm = input.sum(dim=1, keepdim=True) # (batch_size, 1, n_bins, n_frames)
    else:
        raise ValueError("Not support {}-dimension".format(n_dim))
    
    mask = input / (norm + eps) # (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    
    return mask

def wiener_filter_mask(input, eps=EPS):
    """
    Args:
        input (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    n_dim = input.dim()
    power = input**2 # (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    
    if n_dim == 3:
        norm = power.sum(dim=0, keepdim=True) # (1, n_bins, n_frames)
    elif n_dim == 4:
        norm = power.sum(dim=1, keepdim=True) # (batch_size, 1, n_bins, n_frames)
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
        input (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
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

    estimated_spectrgram = estimated_amplitude / amplitude_mixture * spectrogram_mixture
    estimated_signal = torch.istft(estimated_spectrgram, n_fft=fft_size, hop_length=hop_size, length=T)
    estimated_signal = estimated_signal.detach().cpu()
    
    for signal, tag in zip(estimated_signal, ['man', 'woman']):
        torchaudio.save("data/frequency_mask/{}-estimated_{}.wav".format(tag, method), signal.unsqueeze(dim=0), sample_rate=16000, bits_per_sample=16)

if __name__ == '__main__':
    import os
    from scipy.signal import resample_poly
    import torchaudio
    
    from utils.utils_audio import read_wav, write_wav
    
    os.makedirs("data/frequency_mask", exist_ok=True)
    
    fft_size, hop_size = 1024, 256
    n_basis = 4
    
    """
    source1, sr = read_wav("data/man-44100.wav")
    source1 = resample_poly(source1, up=16000, down=sr)
    torchaudio.save("data/man-16000.wav", source1, sample_rate=16000, bits_per_sample=16)
    """
    source1, sr = torchaudio.load("data/man-16000.wav")
    _, T = source1.size()
    
    """
    source2, sr = read_wav("data/woman-44100.wav")
    source2 = resample_poly(source2, up=16000, down=sr)
    torchaudio.save("data/woman-16000.wav", source2, sample_rate=16000, bits_per_sample=16)
    """
    source2, sr = torchaudio.load("data/woman-16000.wav")
    
    mixture = source1 + source2
    """
    torchaudio.save("data/mixture-16000.wav", mixture, sample_rate=16000, bits_per_sample=16)
    """
    
    spectrogram_mixture = torch.stft(mixture, n_fft=fft_size, hop_length=hop_size, return_complex=True)
    amplitude_mixture = torch.abs(spectrogram_mixture)
    
    spectrogram_source1 = torch.stft(source1, n_fft=fft_size, hop_length=hop_size, return_complex=True)
    amplitude_source1 = torch.abs(spectrogram_source1)
    
    spectrogram_source2 = torch.stft(source2, n_fft=fft_size, hop_length=hop_size, return_complex=True)
    amplitude_source2 = torch.abs(spectrogram_source2)

    amplitude = torch.cat([amplitude_source1, amplitude_source2], dim=0)

    _test('IBM')
    _test('IRM')
    _test('WFM')