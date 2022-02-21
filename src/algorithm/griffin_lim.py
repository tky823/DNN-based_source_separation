import math

import torch
import torch.nn as nn

from utils.audio import build_window

class GriffinLim(nn.Module):
    def __init__(self, n_fft, hop_length=None, window_fn='hann'):
        super().__init__()

        if hop_length is None:
            hop_length = n_fft // 4

        self.n_fft, self.hop_length = n_fft, hop_length

        window = build_window(n_fft, window_fn=window_fn)
        self.window = nn.Parameter(window, requires_grad=False)

    def forward(self, amplitude, phase=None, iteration=10):
        """
        Args:
            amplitude (n_bins, n_frames)
        Returns:
            phase (n_bins, n_frames): Reconstructed phase
        """
        for idx in range(iteration):
            phase = self.update(amplitude, phase)

        return phase

    def update(self, amplitude, phase=None):
        """
        Args:
            amplitude (n_bins, n_frames) or (in_channels, n_bins, n_frames) or (in_channels, n_bins, n_frames)
        Returns:
            phase (n_bins, n_frames) or (in_channels, n_bins, n_frames) or (in_channels, n_bins, n_frames)
        """
        n_fft, hop_length = self.n_fft, self.hop_length
        window = self.window

        if torch.is_complex(amplitude):
            raise ValueError("amplitude is NOT expected complex tensor.")

        n_dims = amplitude.dim()

        if n_dims == 2:
            channels = None
            amplitude = amplitude.unsqueeze(dim=0)
        elif n_dims > 2:
            channels = amplitude.size()[:-2]
            amplitude = amplitude.view(-1, *amplitude.size()[-2:])
        else:
            raise ValueError("Invalid shape of tensor.")

        if phase is None:
            sampler = torch.distributions.uniform.Uniform(0, 2*math.pi)
            phase = sampler.sample(amplitude.size()).to(amplitude.device)

        spectrogram = amplitude * torch.exp(1j * phase)

        signal = torch.istft(spectrogram, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=False)
        spectrogram = torch.stft(signal, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=True)
        phase = torch.angle(spectrogram)

        if n_dims == 2:
            phase = phase.squeeze(dim=0)
        elif n_dims > 2:
            phase = phase.view(*channels, *phase.size()[-2:])
        else:
            raise ValueError("Invalid shape of tensor.")

        return phase

class FastGriffinLim(GriffinLim):
    def __init__(self, n_fft, hop_length=None, window_fn='hann'):
        super().__init__(n_fft, hop_length=hop_length, window_fn=window_fn)

        raise NotImplementedError("Coming soon.")

def _test():
    target_sr = 16000
    n_fft, hop_length = 4096, 1024

    signal, sr = torchaudio.load("data/man-44100.wav")
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    signal = resampler(signal)
    torchaudio.save("data/man-{}.wav".format(target_sr), signal, sample_rate=target_sr, bits_per_sample=16)

    T = signal.size(-1)
    window = build_window(n_fft, window_fn='hann')

    spectrogram = torch.stft(signal, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=True)
    oracle_signal = torch.istft(spectrogram, n_fft, hop_length=hop_length, window=window, length=T, onesided=True, return_complex=False)
    torchaudio.save("data/man-oracle-{}.wav".format(target_sr), oracle_signal, sample_rate=target_sr, bits_per_sample=16)

    amplitude = torch.abs(spectrogram)
    griffin_lim = GriffinLim(n_fft, hop_length=hop_length, window_fn='hann')

    # Griffin-Lim iteration 10
    iteration = 10
    estimated_phase = griffin_lim(amplitude, iteration=iteration)
    estimated_spectrogram = amplitude * torch.exp(1j * estimated_phase)
    estimated_signal = torch.istft(estimated_spectrogram, n_fft, hop_length=hop_length, window=window, length=T, onesided=True, return_complex=False)
    torchaudio.save("data/GriffinLim/man-estimated-{}_iter{}.wav".format(target_sr, iteration), estimated_signal, sample_rate=target_sr, bits_per_sample=16)

    # Griffin-Lim iteration 500
    iteration = 500
    estimated_phase = griffin_lim(amplitude, iteration=iteration)
    estimated_spectrogram = amplitude * torch.exp(1j * estimated_phase)
    estimated_signal = torch.istft(estimated_spectrogram, n_fft, hop_length=hop_length, window=window, length=T, onesided=True, return_complex=False)
    torchaudio.save("data/GriffinLim/man-estimated-{}_iter{}.wav".format(target_sr, iteration), estimated_signal, sample_rate=target_sr, bits_per_sample=16)

if __name__ == '__main__':
    import os

    import torchaudio

    os.makedirs("data/GriffinLim", exist_ok=True)
    torch.manual_seed(111)

    _test()