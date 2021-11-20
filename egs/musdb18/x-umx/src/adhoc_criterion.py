import torch
import torch.nn as nn

from transforms.stft import stft, istft
from criterion.combination import CombinationLoss

EPS = 1e-12

class MultiDomainLoss(nn.Module):
    def  __init__(self, criterion_time, criterion_frequency, weight_time=10, weight_frequency=1, combination=True, n_fft=None, hop_length=None, window=None, normalize=False, **kwargs):
        super().__init__()

        if combination:
            source_dim = kwargs['source_dim']
            min_pair, max_pair = kwargs['min_pair'], kwargs['max_pair']
            self.criterion_time = CombinationLoss(criterion_time, combination_dim=source_dim, min_pair=min_pair, max_pair=max_pair)
            self.criterion_frequency = CombinationLoss(criterion_frequency, combination_dim=source_dim, min_pair=min_pair, max_pair=max_pair)
        else:
            self.criterion_time = criterion_time
            self.criterion_frequency = criterion_frequency

        self.weight_time, self.weight_frequency = weight_time, weight_frequency
        self.n_fft, self.hop_length = n_fft, hop_length
        self.window = nn.Parameter(window, requires_grad=False)
        self.normalize = normalize
    
    def forward(self, input, target, batch_mean=True, per_domain=False):
        """
        Args:
            input <torch.Tensor>: Nonnegative tensor with shape of (batch_size, n_sources, n_mics, n_bins, n_frames)
            target <torch.Tensor>: Complex tensor with shape of (batch_size, n_sources, n_mics, n_bins, n_frames)
            batch_mean <bool>: Compute average along batch dimension.
        Returns:
            mixture: (batch_size, n_mics, n_bins, n_frames)
        """
        weight_time, weight_frequency = self.weight_time, self.weight_frequency
        n_fft, hop_length = self.n_fft, self.hop_length
        window = self.window.to(target.device)
        normalize = self.normalize
        
        if torch.is_complex(input):
            raise ValueError("input should be real.")
        
        if not torch.is_complex(target):
            raise ValueError("target should be complex.")

        target_amplitude = torch.abs(target)
        target_time = istft(target, n_fft=n_fft, hop_length=hop_length, window=window, normalized=normalize, return_complex=False)

        mixture_time = target_time.sum(dim=1, keepdim=True) # (batch_size, 1, n_mics, T)
        mixture = stft(mixture_time, n_fft=n_fft, hop_length=hop_length, window=window, normalized=normalize, return_complex=True)
        mixture_phase = torch.angle(mixture) # (batch_size, n_mics, n_bins, n_frames)

        input_amplitude = input
        input = input_amplitude * torch.exp(1j * mixture_phase) # To complex spectrogram
        input_time = istft(input, n_fft=n_fft, hop_length=hop_length, window=window, normalized=normalize, return_complex=False)

        if weight_time == 0 and weight_frequency == 0:
            raise NotImplementedError("Specify weight.")

        if weight_time == 0:
            loss_time = 0
        else:
            loss_time = self.criterion_time(input_time, target_time, batch_mean=batch_mean)

        if weight_frequency == 0:
            loss_frequency = 0
        else:
            loss_frequency = self.criterion_frequency(input_amplitude, target_amplitude, batch_mean=batch_mean)

        if per_domain:
            raise NotImplementedError("Not support per_domain=True.")
            return loss_time, loss_frequency
        else:
            loss = weight_time * loss_time + weight_frequency * loss_frequency

        return loss

def _test_mdl():
    from utils.utils_audio import build_window

    batch_size = 5
    n_sources = 4
    in_channels, T = 2, 32
    n_fft, hop_length = 8, 2
    window = build_window(n_fft, window_fn='hann')

    input = torch.randn(batch_size, n_sources, in_channels, T)
    target = torch.randn(batch_size, n_sources, in_channels, T)

    input = stft(input, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    target = stft(target, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)

    input_amplitude = torch.abs(input)

    criterion = MultiDomainLoss(combination=True, n_fft=n_fft, hop_length=hop_length, window=window)
    loss = criterion(input_amplitude, target)

    print(loss)

if __name__ == '__main__':
    torch.manual_seed(111)

    print("="*10, "Multi-Domain Loss", "="*10)
    _test_mdl()