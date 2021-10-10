import warnings
import math

import torch

EPS = 1e-12

def compute_ideal_binary_mask(input):
    """
    Args:
        input <torch.Tensor>: Complex or nonnegative tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask <torch.Tensor>: Nonnegative tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    if torch.is_complex(input):
        input = torch.abs(input)

    n_dims = input.dim()
    
    if n_dims == 3:
        n_sources, n_bins, n_frames = input.size()
        
        input = input.permute(1, 2, 0).contiguous()
        flatten_input = input.view(n_bins*n_frames, n_sources)
        flatten_idx = torch.arange(0, n_bins*n_frames*n_sources, n_sources)
        flatten_idx = flatten_idx + flatten_input.argmax(dim=1)
        flatten_mask = torch.zeros(n_bins*n_frames*n_sources)
        flatten_mask[flatten_idx] = 1
        
        mask = flatten_mask.view(n_bins, n_frames, n_sources)
        mask = mask.permute(2, 0, 1).contiguous()
    elif n_dims == 4:
        batch_size, n_sources, n_bins, n_frames = input.size()
        
        input = input.permute(0, 2, 3, 1).contiguous()
        flatten_input = input.view(batch_size*n_bins*n_frames, n_sources)
        flatten_idx = torch.arange(0, batch_size*n_bins*n_frames*n_sources, n_sources)
        flatten_idx = flatten_idx + flatten_input.argmax(dim=1)
        flatten_mask = torch.zeros(batch_size*n_bins*n_frames*n_sources)
        flatten_mask[flatten_idx] = 1
        
        mask = flatten_mask.view(batch_size, n_bins, n_frames, n_sources)
        mask = mask.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError("Not support {}-dimension".format(n_dims))
    
    return mask
   
def compute_ideal_ratio_mask(input, eps=EPS):
    """
    Args:
        input <torch.Tensor>: Complex or nonnegative tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask <torch.Tensor>: Nonnegative tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    if torch.is_complex(input):
        input = torch.abs(input)
    
    n_dims = input.dim()
    
    if n_dims == 3:
        norm = input.sum(dim=0, keepdim=True) # (1, n_bins, n_frames)
    elif n_dims == 4:
        norm = input.sum(dim=1, keepdim=True) # (batch_size, 1, n_bins, n_frames)
    else:
        raise ValueError("Not support {}-dimension".format(n_dims))
    
    mask = input / (norm + eps) # (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    
    return mask

def compute_wiener_filter_mask(input, domain=1, eps=EPS):
    """
    Args:
        input <torch.Tensor>: Complex or nonnegative tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
        domain <float>: 1: amplitude, 2: power
    Returns:
        mask <torch.Tensor>: Nonnegative tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    if torch.is_complex(input):
        input = torch.abs(input)
    
    n_dims = input.dim()
    power = input**(2 / domain) # (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    
    if n_dims == 3:
        norm = power.sum(dim=0, keepdim=True) # (1, n_bins, n_frames)
    elif n_dims == 4:
        norm = power.sum(dim=1, keepdim=True) # (batch_size, 1, n_bins, n_frames)
    else:
        raise ValueError("Not support {}-dimension".format(n_dims))
    
    mask = power / (norm + eps)

    return mask

def compute_ideal_amplitude_mask(input, eps=EPS):
    """
    Args:
        input <torch.Tensor>: Complex tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask <torch.Tensor>: Tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    n_dims = input.dim()

    if n_dims == 3:
        mixture = input.sum(dim=0, keepdim=True)
    elif n_dims == 4:
        mixture = input.sum(dim=1, keepdim=True)
    else:
        raise ValueError("3-D or 4-D input is accepted, but given {}.".format(n_dims))
    
    mask = torch.abs(input) / (torch.abs(mixture) + eps)

    return mask

"""
Phase sensitive mask
See "Phase-Sensitive and Recognition-Boosted Speech Separation using Deep Recurrent Neural Networks"
"""
def compute_phase_sensitive_mask(input, eps=EPS):
    """
    Args:
        input <torch.Tensor>: Complex tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask <torch.Tensor>: Tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    n_dims = input.dim()

    if n_dims == 3:
        mixture = input.sum(dim=0, keepdim=True)
    elif n_dims == 4:
        mixture = input.sum(dim=1, keepdim=True)
    else:
        raise ValueError("3-D or 4-D input is accepted, but given {}.".format(n_dims))
    
    angle_mixture, angle_input = torch.angle(mixture), torch.angle(input)
    angle = angle_mixture - angle_input

    mask = (torch.abs(input) / (torch.abs(mixture) + eps)) * torch.cos(angle)
    
    return mask

def compute_ideal_complex_mask(input, eps=EPS):
    """
    Args:
        input <torch.Tensor>: Complex tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    Returns:
        mask <torch.Tensor>: Tensor with shape of (n_sources, n_bins, n_frames) or (batch_size, n_sources, n_bins, n_frames)
    """
    n_dims = input.dim()

    if n_dims == 3:
        mixture = input.sum(dim=0, keepdim=True)
    elif n_dims == 4:
        mixture = input.sum(dim=1, keepdim=True)
    else:
        raise ValueError("3-D or 4-D input is accepted, but given {}.".format(n_dims))
    
    angle = torch.angle(mixture)
    denominator = (torch.abs(mixture) + eps) * torch.exp(1j * angle)
    mask = input / denominator

    return mask

def ideal_binary_mask(input):
    warnings.warn("Use compute_ideal_binary_mask instead.", DeprecationWarning)
    mask = compute_ideal_binary_mask(input)
    return mask

def ideal_ratio_mask(input, eps=EPS):
    warnings.warn("Use compute_ideal_ratio_mask instead.", DeprecationWarning)
    mask = compute_ideal_ratio_mask(input, eps=eps)
    return mask

def wiener_filter_mask(input, domain=1, eps=EPS):
    warnings.warn("Use compute_wiener_filter_mask instead.", DeprecationWarning)
    mask = compute_wiener_filter_mask(input, domain=domain, eps=eps)
    return mask

def ideal_amplitude_mask(input, eps=EPS):
    warnings.warn("Use compute_ideal_amplitude_mask instead.", DeprecationWarning)
    mask = compute_ideal_amplitude_mask(input, eps=eps)
    return mask

def phase_sensitive_mask(input, eps=EPS):
    warnings.warn("Use compute_phase_sensitive_mask instead.", DeprecationWarning)
    mask = compute_phase_sensitive_mask(input, eps=eps)
    return mask

def ideal_complex_mask(input, eps=EPS):
    warnings.warn("Use compute_ideal_complex_mask instead.", DeprecationWarning)
    mask = compute_ideal_complex_mask(input, eps=eps)
    return mask

def multichannel_wiener_filter(mixture, estimated_sources_amplitude, iteration=1, channels_first=True, eps=EPS):
    """
    Multichannel Wiener filter.
    Implementation is based on norbert package.
    Args:
        mixture <torch.Tensor>: Complex tensor with shape of (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames) or (batch_size, 1, n_channels, n_bins, n_frames) or (batch_size, n_channels, n_bins, n_frames)
        estimated_sources_amplitude <torch.Tensor>: Nonnegative tensor with shape of (n_sources, n_channels, n_bins, n_frames) or (batch_size, n_sources, n_channels, n_bins, n_frames)
        iteration <int>: Iteration of EM algorithm updates
        channels_first <bool>: Only supports True
        eps <float>: small value for numerical stability
    """
    assert channels_first, "`channels_first` is expected True, but given {}".format(channels_first)

    n_dims = estimated_sources_amplitude.dim()
    n_dims_mixture = mixture.dim()

    if n_dims == 4:
        """
        Shape of mixture is (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames)
        """
        if n_dims_mixture == 4:
            mixture = mixture.squeeze(dim=1) # (n_channels, n_bins, n_frames)
        elif n_dims_mixture != 3:
            raise ValueError("mixture.dim() is expected 3 or 4, but given {}.".format(mixture.dim()))
        
        # Use soft mask
        ratio = estimated_sources_amplitude / (estimated_sources_amplitude.sum(dim=0) + eps)
        estimated_sources = ratio * mixture

        norm = max(1, torch.abs(mixture).max() / 10)
        mixture, estimated_sources = mixture / norm, estimated_sources / norm

        estimated_sources = update_em(mixture, estimated_sources, iteration, eps=eps)
        estimated_sources = norm * estimated_sources
    elif n_dims == 5:
        """
        Shape of mixture is (batch_size, 1, n_channels, n_bins, n_frames) or (batch_size, n_channels, n_bins, n_frames)
        """
        if n_dims_mixture == 5:
            mixture = mixture.squeeze(dim=1) # (batch_size, n_channels, n_bins, n_frames)
        elif n_dims_mixture != 4:
            raise ValueError("mixture.dim() is expected 4 or 5, but given {}.".format(mixture.dim()))
        
        estimated_sources = []

        for _mixture, _estimated_sources_amplitude in zip(mixture, estimated_sources_amplitude):
            # Use soft mask
            ratio = _estimated_sources_amplitude / (_estimated_sources_amplitude.sum(dim=0) + eps)
            _estimated_sources = ratio * _mixture

            norm = max(1, torch.abs(_mixture).max() / 10)
            _mixture, _estimated_sources = _mixture / norm, _estimated_sources / norm

            _estimated_sources = update_em(_mixture, _estimated_sources, iteration, eps=eps)
            _estimated_sources = norm * _estimated_sources

            estimated_sources.append(_estimated_sources.unsqueeze(dim=0))
        
        estimated_sources = torch.cat(estimated_sources, dim=0)
    else:
        raise ValueError("estimated_sources_amplitude.dim() is expected 4 or 5, but given {}.".format(estimated_sources_amplitude.dim()))

    return estimated_sources

"""
For multichannel Wiener filter
"""
def update_em(mixture, estimated_sources, iteration=1, source_parallel=False, bin_parallel=True, frame_parallel=True, eps=EPS):
    """
    Args:
        mixture: (n_channels, n_bins, n_frames)
        estimated_sources: (n_sources, n_channels, n_bins, n_frames)
    Returns
        estiamted_sources: (n_sources, n_channels, n_bins, n_frames)
    """
    n_sources, n_channels, _, _ = estimated_sources.size()

    for iteration_idx in range(iteration):
        v, R = [], []
        Cxx = 0

        if source_parallel:
            v, R = get_stats(estimated_sources, eps=eps) # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_channels, n_channels)
            Cxx = torch.sum(v.unsqueeze(dim=4) * R, dim=0) # (n_bins, n_frames, n_channels, n_channels)
        else:
            for source_idx in range(n_sources):
                y_n = estimated_sources[source_idx] # (n_channels, n_bins, n_frames)
                v_n, R_n = get_stats(y_n, eps=eps) # (n_bins, n_frames), (n_bins, n_channels, n_channels)
                Cxx = Cxx + v_n.unsqueeze(dim=2).unsqueeze(dim=3) * R_n.unsqueeze(dim=1) # (n_bins, n_frames, n_channels, n_channels)
                v.append(v_n.unsqueeze(dim=0))
                R.append(R_n.unsqueeze(dim=0))
        
            v, R = torch.cat(v, dim=0), torch.cat(R, dim=0) # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_channels, n_channels)
       
        v, R = v.unsqueeze(dim=3), R.unsqueeze(dim=2) # (n_sources, n_bins, n_frames, 1), (n_sources, n_bins, 1, n_channels, n_channels)

        if bin_parallel:
            if frame_parallel:
                inv_Cxx = torch.linalg.inv(Cxx + math.sqrt(eps) * torch.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)
            else:
                n_frames = Cxx.size(1)

                inv_Cxx = []
                for frame_idx in range(n_frames):
                    _Cxx = Cxx[:, frame_idx]
                    _inv_Cxx = torch.linalg.inv(_Cxx + math.sqrt(eps) * torch.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)
                    inv_Cxx.append(_inv_Cxx)
                
                inv_Cxx = torch.stack(inv_Cxx, dim=1)
        else:
            inv_Cxx = []
            if frame_parallel:
                for _Cxx in Cxx:
                    _inv_Cxx = torch.linalg.inv(_Cxx + math.sqrt(eps) * torch.eye(n_channels)) # (n_frames, n_channels, n_channels)
                    inv_Cxx.append(_inv_Cxx)
            else:
                for _Cxx in Cxx:
                    _inv_Cxx = []
                    for __Cxx in _Cxx:
                        __inv_Cxx = torch.linalg.inv(__Cxx + math.sqrt(eps) * torch.eye(n_channels)) # (n_channels, n_channels)
                        _inv_Cxx.append(__inv_Cxx)
                    _inv_Cxx = torch.stack(_inv_Cxx, dim=0)
                    inv_Cxx.append(_inv_Cxx)

            inv_Cxx = torch.stack(inv_Cxx, dim=0)

        if source_parallel:
            gain = v.unsqueeze(dim=4) * torch.sum(R.unsqueeze(dim=5) * inv_Cxx.unsqueeze(dim=2), dim=4) # (n_sources, n_bins, n_frames, n_channels, n_channels)
            gain = gain.permute(0, 3, 4, 1, 2) # (n_sources, n_channels, n_channels, n_bins, n_frames)
            estimated_sources = torch.sum(gain * mixture, dim=2) # (n_sources, n_channels, n_bins, n_frames)
        else:
            estimated_sources = []

            for source_idx in range(n_sources):
                v_n, R_n = v[source_idx], R[source_idx] # (n_bins, n_frames, 1), (n_bins, 1, n_channels, n_channels)

                gain_n = v_n.unsqueeze(dim=3) * torch.sum(R_n.unsqueeze(dim=4) * inv_Cxx.unsqueeze(dim=2), dim=3) # (n_bins, n_frames, n_channels, n_channels)
                gain_n = gain_n.permute(2, 3, 0, 1) # (n_channels, n_channels, n_bins, n_frames)
                estimated_source = torch.sum(gain_n * mixture, dim=1) # (n_channels, n_bins, n_frames)
                estimated_sources.append(estimated_source.unsqueeze(dim=0))
            
            estimated_sources = torch.cat(estimated_sources, dim=0) # (n_sources, n_channels, n_bins, n_frames)

    return estimated_sources

def get_stats(spectrogram, eps=EPS):
    """
    Compute empirical parameters of local gaussian model.
    Args:
        spectrogram <torch.Tensor>: (n_mics, n_bins, n_frames) or (n_sources, n_mics, n_bins, n_frames)
    Returns:
        psd <torch.Tensor>: (n_bins, n_frames) or (n_sources, n_bins, n_frames)
        covariance <torch.Tensor>: (n_bins, n_frames, n_mics, n_mics) or (n_sources, n_bins, n_frames, n_mics, n_mics)
    """
    n_dims = spectrogram.dim()

    if n_dims == 3:
        psd = torch.mean(torch.abs(spectrogram)**2, dim=0) # (n_bins, n_frames)
        covariance = spectrogram.unsqueeze(dim=1) * spectrogram.unsqueeze(dim=0).conj() # (n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.sum(dim=3) # (n_mics, n_mics, n_bins)
        denominator = psd.sum(dim=1) + eps # (n_bins,)

        covariance = covariance / denominator # (n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.permute(2, 0, 1) # (n_bins, n_mics, n_mics)
    elif n_dims == 4:
        psd = torch.mean(torch.abs(spectrogram)**2, dim=1) # (n_sources, n_bins, n_frames)
        covariance = spectrogram.unsqueeze(dim=2) * spectrogram.unsqueeze(dim=1).conj() # (n_sources, n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.sum(dim=4) # (n_sources, n_mics, n_mics, n_bins)
        denominator = psd.sum(dim=2) + eps # (n_sources, n_bins)
        
        covariance = covariance / denominator.unsqueeze(dim=1).unsqueeze(dim=2) # (n_sources, n_mics, n_mics, n_bins)
        covariance = covariance.permute(0, 3, 1, 2) # (n_sources, n_bins, n_mics, n_mics)
    else:
        raise ValueError("Invalid dimension of tensor is given.")

    return psd, covariance

def _test_amplitude(amplitude, method='IBM'):
    if method == 'IBM':
        mask = compute_ideal_binary_mask(amplitude)
    elif method == 'IRM':
        mask = compute_ideal_ratio_mask(amplitude)
    elif method == 'WFM':
        mask = compute_wiener_filter_mask(amplitude)
    else:
        raise NotImplementedError("Not support {}".format(method))
    
    estimated_amplitude = amplitude * mask
    estimated_spectrgram = estimated_amplitude * torch.exp(1j * torch.angle(spectrogram_mixture))
    estimated_signal = torch.istft(estimated_spectrgram, n_fft=fft_size, hop_length=hop_size, length=T)
    estimated_signal = estimated_signal.detach().cpu()
    
    for signal, tag in zip(estimated_signal, ['man', 'woman']):
        torchaudio.save("data/frequency_mask/{}-estimated_{}.wav".format(tag, method), signal.unsqueeze(dim=0), sample_rate=16000, bits_per_sample=16)

def _test_spectrogram(spectrogram, method='PSM'):
    if method == 'IAM':
        mask = compute_ideal_amplitude_mask(spectrogram)
    elif method == 'PSM':
        mask = compute_phase_sensitive_mask(spectrogram)
    elif method == 'ICM':
        mask = compute_ideal_complex_mask(spectrogram)
    else:
        raise NotImplementedError("Not support {}".format(method))
    
    estimated_amplitude = amplitude * mask
    estimated_spectrgram = estimated_amplitude * torch.exp(1j * torch.angle(spectrogram_mixture))
    estimated_signal = torch.istft(estimated_spectrgram, n_fft=fft_size, hop_length=hop_size, length=T)
    estimated_signal = estimated_signal.detach().cpu()
    
    for signal, tag in zip(estimated_signal, ['man', 'woman']):
        torchaudio.save("data/frequency_mask/{}-estimated_{}.wav".format(tag, method), signal.unsqueeze(dim=0), sample_rate=16000, bits_per_sample=16)

if __name__ == '__main__':
    import os

    import torchaudio
    
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

    spectrogram = torch.cat([spectrogram_source1, spectrogram_source2], dim=0)
    amplitude = torch.cat([amplitude_source1, amplitude_source2], dim=0)

    _test_amplitude(amplitude, method='IBM')
    _test_amplitude(amplitude, method='IRM')
    _test_amplitude(amplitude, method='WFM')

    _test_spectrogram(spectrogram, method='IAM')
    _test_spectrogram(spectrogram, method='PSM')
    _test_spectrogram(spectrogram, method='ICM')