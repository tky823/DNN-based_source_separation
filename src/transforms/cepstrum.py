import torch

from transforms.stft import stft

EPS = 1e-12

def real_cepstrum(input, n_fft=None, minimum_distortion=False, eps=EPS):
    if torch.is_complex(input):
        raise ValueError("input should be real.")

    if n_fft is None:
        n_fft = input.size(-1)

    cepstrum = torch.fft.irfft(torch.log(torch.abs(torch.fft.rfft(input, n_fft)) + eps), n_fft)

    if minimum_distortion:
        odd = n_fft % 2
        ones_left, ones_center, ones_right = torch.ones(1), torch.ones((n_fft + odd) // 2 - 1), torch.ones(1 - odd)
        zeros = torch.zeros((n_fft + odd) // 2 - 1)
        window = torch.cat([ones_left, 2 * ones_center, ones_right, zeros])
        window = window.to(cepstrum.device)
        cepstrum = torch.fft.irfft(torch.exp(torch.fft.rfft(window * cepstrum, n_fft)), n_fft)

    return cepstrum

def compute_cepsptrogram(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, eps=EPS):
    """
    Args:
        input <torch.Tensor>: (*, T)
    Returns:
        output <torch.Tensor>: Cepstrogram with shape of (*, n_bins, n_frames), where n_bins = n_fft // 2 + 1
    """
    if torch.is_complex(input):
        raise ValueError("input should be real.")

    assert not normalized, "normalized is expected to be False."

    spectrogram = stft(input, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, normalized=normalized, onesided=True, return_complex=True)
    output = spectrogram_to_cepsptrogram(spectrogram, n_fft=n_fft, domain=1, onesided=True, eps=eps)

    return output

def spectrogram_to_cepsptrogram(input, n_fft=None, domain=1, onesided=True, eps=EPS):
    """
    Args:
        input <torch.Tensor>: Spectrogram with shape of (*, n_bins, n_frames).
    Returns:
        output <torch.Tensor>: Cepstrogram with shape of (*, n_bins, n_frames).
    """
    assert onesided, "onesided should be True."

    if torch.is_complex(input):
        assert domain == 1, "domain should be 1."
        amplitude_spectrogram = torch.abs(input)
    else:
        amplitude_spectrogram = input ** (1 / domain)

    if n_fft is None:
        n_bins = input.size(-2)
        n_fft = 2 * (n_bins - 1)
    else:
        n_bins = n_fft // 2 + 1

    log_amplitude_spectrogram = torch.log(amplitude_spectrogram + eps)
    cepsptrogram = torch.fft.irfft(log_amplitude_spectrogram, n_fft, dim=-2)

    sections = [n_bins, n_fft - n_bins]
    output, _ = torch.split(cepsptrogram, sections, dim=-2)

    return output

def cepsptrogram_to_amplitude(input, n_fft=None, onesided=True):
    """
    Args:
        input <torch.Tensor>: Cepstrogram with shape of (*, n_bins, n_frames).
    Returns:
        output <torch.Tensor>: Amplitude spectrogram with shape of (*, n_bins, n_frames).
    """
    assert onesided, "onesided should be True."

    n_bins = input.size(-2)

    if n_fft is None:
        n_fft = 2 * (n_bins - 1)
    else:
        n_bins = n_fft // 2 + 1

    log_amplitude_spectrogram = torch.fft.irfft(input, n_fft, dim=-2, norm="forward")

    sections = [n_bins, n_fft - n_bins]
    log_amplitude_spectrogram, _ = torch.split(log_amplitude_spectrogram, sections, dim=-2)
    output = torch.exp(log_amplitude_spectrogram)

    return output

def _test_rceps():
    waveform, _ = torchaudio.load("../algorithm/data/single-channel/mtlb.wav")
    cepstrum = real_cepstrum(waveform)
    print(cepstrum)

    minimum_distortion_cepstrum = real_cepstrum(waveform, minimum_distortion=True)
    print(minimum_distortion_cepstrum)

def _test_spectrogram_to_cepstrogram():
    n_fft, hop_length = 256, 64
    window = build_window(n_fft, window_fn="hann")
    waveform, _ = torchaudio.load("../algorithm/data/single-channel/mtlb.wav")

    spectrogram = stft(waveform, n_fft, hop_length=hop_length, window=window, return_complex=True)
    amplitude_spectrogram = torch.abs(spectrogram)
    cepsptrogram = spectrogram_to_cepsptrogram(amplitude_spectrogram, n_fft=n_fft)
    reconstructed_amplitude_spectrogram = cepsptrogram_to_amplitude(cepsptrogram)
    print(torch.allclose(amplitude_spectrogram, reconstructed_amplitude_spectrogram))

def _test_cepstrogram():
    n_fft, hop_length = 256, 64
    window = build_window(n_fft, window_fn="hann")
    waveform, _ = torchaudio.load("../algorithm/data/single-channel/mtlb.wav")

    cepsptrogram = compute_cepsptrogram(waveform, n_fft, hop_length=hop_length, window=window)
    print(waveform.size(), cepsptrogram.size())

def _test_rceps_echo_cancel():
    waveform, sample_rate = torchaudio.load("./data/single-channel/mtlb.wav")

    lag, alpha = 0.23, 0.5
    delta = round(lag * sample_rate)

    silence = torch.zeros(waveform.size(0), delta)
    orig = torch.cat([waveform, silence], dim=1)
    echo = alpha * torch.cat([silence, waveform], dim=1)
    reverbed = orig + echo

    cepstrum = real_cepstrum(reverbed)
    print(cepstrum)

    plt.figure()
    plt.plot(cepstrum.squeeze())
    plt.show()
    plt.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchaudio

    from utils.audio import build_window

    _test_rceps()
    print()

    _test_spectrogram_to_cepstrogram()
    print()

    _test_cepstrogram()
    print()

    # _test_rceps_echo_cancel()