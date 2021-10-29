import itertools

import torch
import torch.nn as nn

EPS = 1e-12

class MultiDomainLoss(nn.Module):
    def  __init__(self, criterion_time, criterion_frequency, weight_time=10, weight_frequency=1, combination=True, fft_size=None, hop_size=None, window=None, normalize=False):
        super().__init__()

        if combination:
            self.criterion_time = CombinationLoss(criterion_time)
            self.criterion_frequency = CombinationLoss(criterion_frequency)
        else:
            self.criterion_time = criterion_time
            self.criterion_frequency = criterion_frequency

        self.weight_time, self.weight_frequency = weight_time, weight_frequency
        self.fft_size, self.hop_size = fft_size, hop_size
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
        fft_size, hop_size = self.fft_size, self.hop_size
        window = self.window.to(target.device)
        normalize = self.normalize
        
        if torch.is_complex(input):
            raise ValueError("input should be real.")
        
        if not torch.is_complex(target):
            raise ValueError("target should be complex.")
        
        batch_size, n_sources, n_mics, n_bins, n_frames = target.size()

        target_amplitude = torch.abs(target)
        target = target.view(batch_size * n_sources * n_mics, n_bins, n_frames)
        target_time = torch.istft(target, n_fft=fft_size, hop_length=hop_size, window=window, normalized=normalize, return_complex=False)
        target_time = target_time.view(batch_size, n_sources, n_mics, -1)

        mixture_time = target_time.sum(dim=1, keepdim=True) # (batch_size, 1, n_mics, T)
        mixture_time = mixture_time.view(batch_size * n_mics, -1)
        mixture = torch.stft(mixture_time, n_fft=fft_size, hop_length=hop_size, window=window, normalized=normalize, return_complex=True)
        mixture_phase = torch.angle(mixture) # (batch_size * n_mics, n_bins, n_frames)
        mixture_phase = mixture_phase.view(batch_size, 1, n_mics, n_bins, n_frames)

        input_amplitude = input
        input = input_amplitude * torch.exp(1j * mixture_phase) # To complex spectrogram
        input = input.view(batch_size * n_sources * n_mics, n_bins, n_frames)
        input_time = torch.istft(input, n_fft=fft_size, hop_length=hop_size, window=window, normalized=normalize, return_complex=False)
        input_time = input_time.view(batch_size, n_sources, n_mics, -1) # (batch_size, n_sources, n_mics, T)

        if weight_time == 0 and weight_frequency == 0:
            raise NotImplementedError("Specify weight.")

        if weight_time == 0:
            loss_time = 0
        else:
            print("time", input_time.size(), target_time.size(), end='->')
            loss_time = self.criterion_time(input_time, target_time, batch_mean=batch_mean)
            print(loss_time.size(), flush=True)

        if weight_frequency == 0:
            loss_frequency = 0
        else:
            print("frequency", input_amplitude.size(), target_amplitude.size(), end='->')
            loss_frequency = self.criterion_frequency(input_amplitude, target_amplitude, batch_mean=batch_mean)
            print(loss_frequency.size(), flush=True)

        if per_domain:
            raise NotImplementedError("Not support return_separately=True.")
            return loss_time, loss_frequency
        else:
            loss = weight_time * loss_time + weight_frequency * loss_frequency

        return loss

class CombinationLoss(nn.Module):
    """
    Combination Loss for Multi Sources
    """
    def __init__(self, criterion, source_dim=1, min_pair=1, max_pair=None):
        super().__init__()

        self.criterion = criterion

        self.source_dim = source_dim
        self.min_pair, self.max_pair = min_pair, max_pair

    def forward(self, input, target, combination_mean=True, batch_mean=True):
        assert target.size() == input.size(), "input.size() are expected same."

        source_dim = self.source_dim
        min_pair, max_pair = self.min_pair, self.max_pair

        n_sources = input.size(source_dim)

        if max_pair is None:
            max_pair = n_sources - 1
        
        input = torch.unbind(input, dim=source_dim)
        target = torch.unbind(target, dim=source_dim)

        loss = []

        for _n_sources in range(min_pair, max_pair + 1):
            for pair_indices in itertools.combinations(range(n_sources), _n_sources):
                _input, _target = [], []
                for idx in pair_indices:
                    _input.append(input[idx])
                    _target.append(target[idx])
                _input, _target = torch.stack(_input, dim=0), torch.stack(_target, dim=0)
                _input, _target = _input.sum(dim=0), _target.sum(dim=0)

                loss_pair = self.criterion(_input, _target, batch_mean=batch_mean)
                loss.append(loss_pair)

        loss = torch.stack(loss, dim=0)

        if combination_mean:
            loss = loss.mean(dim=0)
        else:
            loss = loss.sum(dim=0)
        
        return loss

def _test_mdl():
    from utils.utils_audio import build_window

    batch_size = 5
    n_sources = 4
    in_channels, T = 2, 32
    fft_size, hop_size = 8, 2
    window = build_window(fft_size, window_fn='hann')

    input = torch.randn(batch_size, n_sources, in_channels, T)
    target = torch.randn(batch_size, n_sources, in_channels, T)

    input = input.view(batch_size * n_sources * in_channels, T)
    target = target.view(batch_size * n_sources * in_channels, T)

    input = torch.stft(input, n_fft=fft_size, hop_length=hop_size, window=window, return_complex=True)
    target = torch.stft(target, n_fft=fft_size, hop_length=hop_size, window=window, return_complex=True)

    input = input.view(batch_size, n_sources, in_channels, *input.size()[-2:])
    target = target.view(batch_size, n_sources, in_channels, *target.size()[-2:])

    input_amplitude = torch.abs(input)

    criterion = MultiDomainLoss(combination=True, fft_size=fft_size, hop_size=hop_size, window=window)
    loss = criterion(input_amplitude, target)

    print(loss)

def _test_cl():
    batch_size = 3
    n_sources = 4
    in_channels, T = 2, 32

    input = torch.randn(batch_size, n_sources, in_channels, T)
    target = torch.randn(batch_size, n_sources, in_channels, T)

    criterion = NegWeightedSDR()
    combination_criterion = CombinationLoss(criterion, min_pair=1, max_pair=n_sources-1)

    loss = combination_criterion(input, target, batch_mean=False)
    print(loss)

if __name__ == '__main__':
    from criterion.sdr import NegWeightedSDR

    torch.manual_seed(111)

    print("="*10, "Combination Loss", "="*10)
    _test_cl()
    print()

    print("="*10, "Multi-Domain Loss", "="*10)
    _test_mdl()