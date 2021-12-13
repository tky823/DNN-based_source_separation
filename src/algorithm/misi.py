import torch
import torch.nn as nn

from utils.audio import build_window

"""
Multiple input spectrogram inverse
Reference "Iterative Phase Estimation for the Synthesis of Separated Sources from Single-Channel Mixtures"
"""

EPS = 1e-12

class MISI(nn.Module):
    def __init__(self, n_fft, hop_length=None, window=None, window_fn=None):
        super().__init__()

        if hop_length is None:
            hop_length = n_fft // 2

        self.n_fft, self.hop_length = n_fft, hop_length

        if window is not None:
            if window_fn is not None:
                raise ValueError("Specify either window or window_fn")
        else:
            window = build_window(n_fft, window_fn=window_fn)

        if not isinstance(window, nn.Parameter):
            window = nn.Parameter(window, requires_grad=False)

        self.window = window

    def forward(self, mixture, estimated_sources_amplitude, iteration=10, return_all_iterations=False, iteration_dim=0):
        """
        Args:
            mixture <torch.Tensor>: Comlex spectrogram with shape of (batch_size, 1, n_bins, n_frames).
            estimated_sources_amplitude <torch.Tensor>: Amplitude spectrogram with shape of (batch_size, n_sources, n_bins, n_frames).
        Returns:
            estimated_sources <torch.Tensor>:
                Comlex spectrogram with shape of (batch_size, n_sources, n_bins, n_frames) if return_all_iterations=False (default).
                Comlex spectrogram with shape of (iteration, batch_size, n_sources, n_bins, n_frames) if return_all_iterations=True. You can set different dimension for iteration using iteration_dim.
        """
        if not torch.is_complex(mixture):
            raise TypeError("mixture is expected complex tensor.")

        if torch.is_complex(estimated_sources_amplitude):
            raise TypeError("estimated_sources_amplitude is expected complex tensor.")

        n_fft, hop_length = self.n_fft, self.hop_length
        window = self.window

        phase = torch.angle(mixture)
        estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase)

        mixture_channels = mixture.size()[:-2]
        mixture = mixture.view(-1, *mixture.size()[-2:])
        mixture = torch.istft(mixture, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=False)
        mixture = mixture.view(*mixture_channels, *mixture.size()[-1:])

        estimated_sources_all_iterations = []
        for idx in range(iteration):
            phase = self.update_phase_once(mixture=mixture, estimated_sources=estimated_sources)
            estimated_sources = estimated_sources_amplitude * torch.exp(1j * phase)
            if return_all_iterations:
                estimated_sources_all_iterations.append(estimated_sources)

        if return_all_iterations:
            estimated_sources = torch.stack(estimated_sources_all_iterations, dim=iteration_dim)

        return estimated_sources

    def update_phase_once(self, mixture, estimated_sources):
        """
        Args:
            mixture <torch.Tensor>: Time domain signal with shape of (batch_size, 1, T).
            estimated_sources_amplitude <torch.Tensor>: Complex spectrogram with shape of (batch_size, n_sources, n_bins, n_frames).
        """
        n_fft, hop_length = self.n_fft, self.hop_length
        window = self.window

        _, n_sources, _, _ = estimated_sources.size()

        estimated_sources_channels = estimated_sources.size()[:-2]
        estimated_sources = estimated_sources.view(-1, *estimated_sources.size()[-2:])
        estimated_sources = torch.istft(estimated_sources, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=False)
        estimated_sources = estimated_sources.view(*estimated_sources_channels, *estimated_sources.size()[-1:])

        delta = mixture - torch.sum(estimated_sources, dim=1, keepdim=True)

        estimated_sources = estimated_sources + delta / n_sources

        estimated_sources = estimated_sources.view(-1, estimated_sources.size(-1))
        estimated_sources = torch.stft(estimated_sources, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=True)
        estimated_sources = estimated_sources.view(*estimated_sources_channels, *estimated_sources.size()[-2:])

        estimated_sources_phase = torch.angle(estimated_sources)

        return estimated_sources_phase

def _test_danet():
    n_sources = 2
    sr = 8000
    n_fft, hop_length = 256, 64
    threshold = 40
    iter_clustering = 10
    exp_dir = "../../egs/tutorials/danet/exp"
    model_path = os.path.join(exp_dir, "2mix/l2loss/stft256-64_hamming-window_ibm_threshold40/K20_H256_B4_causal0_mask-sigmoid/b4_e100_rmsprop-lr1e-4-decay0/seed111/model/last.pth")

    mixture, sr = torchaudio.load("./data/mixture-{}.wav".format(sr))
    window = build_window(n_fft, window_fn='hann')
    mixture = torch.stft(mixture, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=True)

    mixture = mixture.unsqueeze(dim=0)
    mixture_amplitude = torch.abs(mixture)
    log_amplitude = 20 * torch.log10(mixture_amplitude + EPS)
    max_log_amplitude = torch.max(log_amplitude)
    threshold = 10**((max_log_amplitude - threshold) / 20)
    threshold_weight = torch.where(mixture_amplitude > 0, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))

    model = DANet.build_model(model_path, load_state_dict=True)
    model.eval()

    with torch.no_grad():
        estimated_sources_amplitude = model(mixture_amplitude, assignment=None, threshold_weight=threshold_weight, n_sources=n_sources, iter_clustering=iter_clustering)

    misi = MISI(n_fft=n_fft, hop_length=hop_length, window_fn='hann')

    # Iteration 0
    iteration = 0
    estimated_sources = misi(mixture, estimated_sources_amplitude=estimated_sources_amplitude, iteration=iteration)
    estimated_sources = estimated_sources.squeeze(dim=0)
    estimated_sources = torch.istft(estimated_sources, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=False)

    for idx, estimated_source in enumerate(estimated_sources):
        estimated_source = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
        torchaudio.save("./data/MISI/estimated-{}-{}_iter{}.wav".format(sr, idx + 1, iteration), estimated_source, sample_rate=sr)

    # Iteration 10
    iteration = 10
    estimated_sources = misi(mixture, estimated_sources_amplitude=estimated_sources_amplitude, iteration=iteration)
    estimated_sources = estimated_sources.squeeze(dim=0)
    estimated_sources = torch.istft(estimated_sources, n_fft, hop_length=hop_length, window=window, onesided=True, return_complex=False)

    for idx, estimated_source in enumerate(estimated_sources):
        estimated_source = estimated_source.unsqueeze(dim=0) if estimated_source.dim() == 1 else estimated_source
        torchaudio.save("./data/MISI/estimated-{}-{}_iter{}.wav".format(sr, idx + 1, iteration), estimated_source, sample_rate=sr)

def _downsample():
    sr, target_sr = 16000, 8000
    signal, sr = torchaudio.load("data/mixture-{}.wav".format(sr))
    resampler = torchaudio.transforms.Resample(sr, target_sr)
    signal = resampler(signal)
    torchaudio.save("data/mixture-{}.wav".format(target_sr), signal, sample_rate=target_sr, bits_per_sample=16)

if __name__ == '__main__':
    import os

    import torchaudio

    from models.danet import DANet

    os.makedirs("data/MISI", exist_ok=True)
    torch.manual_seed(111)

    # _downsample()
    _test_danet()