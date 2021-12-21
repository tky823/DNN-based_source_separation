import torch

def stft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode='reflect', normalized=False, onesided=None, return_complex=None):
    """
    Args:
        input <torch.Tensor>: (*, length)
    Returns:
        output <torch.Tensor>:
            (*, n_bins, n_frames), where n_bins = n_fft // 2 + 1 if onesided = True and return_complex = True.
            (*, n_bins, n_frames, 2), where n_bins = n_fft // 2 + 1 if onesided = True and return_complex = False.
    """
    n_dims = input.dim()

    if n_dims == 1:
        input = input.unsqueeze(dim=0)
    elif n_dims > 2:
        channels = input.size()[:-1]
        input = input.view(-1, input.size(-1))

    output = torch.stft(
        input,
        n_fft, hop_length=hop_length, win_length=win_length, window=window,
        center=center, pad_mode=pad_mode, normalized=normalized,
        onesided=onesided, return_complex=return_complex
    )

    if return_complex:
        if n_dims == 1:
            output = output.squeeze(dim=0)
        elif n_dims > 2:
            output = output.view(*channels, *output.size()[-2:])
    else:
        if n_dims == 1:
            output = output.squeeze(dim=0)
        elif n_dims > 2:
            output = output.view(*channels, *output.size()[-3:])

    return output

def istft(input, n_fft, hop_length=None, win_length=None, window=None, center=True, normalized=False, onesided=None, length=None, return_complex=False):
    """
    Args:
        input <torch.Tensor>: Complex tensor with shape of
            (*, n_bins, n_frames), where n_bins = n_fft // 2 + 1
    Returns:
        output <torch.Tensor>: (*, length)
    """
    if not torch.is_complex(input):
        raise TypeError("Not support real input.")

    n_dims = input.dim()

    if n_dims == 2:
        input = input.unsqueeze(dim=0)
    elif n_dims > 3:
        channels = input.size()[:-2]
        input = input.view(-1, *input.size()[-2:])

    output = torch.istft(
        input,
        n_fft, hop_length=hop_length, win_length=win_length, window=window,
        center=center, normalized=normalized,
        onesided=onesided, length=length, return_complex=return_complex
    )

    if n_dims == 2:
        output = output.squeeze(dim=0)
    elif n_dims > 3:
        output = output.view(*channels, -1)

    return output

def _test_stft_istft():
    batch_size, n_channels, T = 4, 2, 10
    n_fft, hop_length = 4, 1
    window = torch.hann_window(n_fft)
    onesided, return_complex = True, True

    waveform_in = torch.randn(T)
    spectrogram = stft(waveform_in, n_fft=n_fft, hop_length=hop_length, window=window, onesided=onesided, return_complex=return_complex)
    waveform_out = istft(spectrogram, n_fft=n_fft, hop_length=hop_length, window=window, onesided=onesided, length=T, return_complex=False)
    print(waveform_in.size(), spectrogram.size(), waveform_out.size())

    waveform_in = torch.randn(batch_size, T)
    spectrogram = stft(waveform_in, n_fft=n_fft, hop_length=hop_length, window=window, onesided=onesided, return_complex=return_complex)
    waveform_out = istft(spectrogram, n_fft=n_fft, hop_length=hop_length, window=window, onesided=onesided, length=T, return_complex=False)
    print(waveform_in.size(), spectrogram.size(), waveform_out.size())

    waveform_in = torch.randn(batch_size, n_channels, T)
    spectrogram = stft(waveform_in, n_fft=n_fft, hop_length=hop_length, window=window, onesided=onesided, return_complex=return_complex)
    waveform_out = istft(spectrogram, n_fft=n_fft, hop_length=hop_length, window=window, onesided=onesided, length=T, return_complex=False)
    print(waveform_in.size(), spectrogram.size(), waveform_out.size())

if __name__ == '__main__':
    torch.manual_seed(111)

    _test_stft_istft()