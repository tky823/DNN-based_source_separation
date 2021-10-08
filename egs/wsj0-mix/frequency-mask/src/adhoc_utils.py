import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_audio import build_window
from algorithm.frequency_mask import ideal_binary_mask, ideal_ratio_mask, phase_sensitive_mask

class FrequencyMasking(nn.Module):
    def __init__(self, fft_size, hop_size=None, window_fn='hann', domain='time'):
        if hop_size is None:
            hop_size = fft_size // 2
        
        self.fft_size, self.hop_size = fft_size, hop_size
        window = build_window(fft_size, window_fn=window_fn)
        self.window = nn.Parameter(window, requires_grad=False)

        assert domain in ['time'], "domain is expected time."

        self.domain = domain

    def forward(self, mixture, sources):
        T = mixture.size(-1)
        mixture, sources = self.stft(mixture), self.stft(sources)

        n_dims = sources.dim()
        mixture_dims = mixture.dim()
        
        if n_dims == 3:
            if mixture_dims == 2:
                mixture = mixture.unsqueeze(dim=0)
                squeeze_dim = 0
            else:
                squeeze_dim = None
            input = sources
        elif n_dims == 4:
            if mixture_dims == 3:
                mixture = mixture.unsqueeze(dim=1)
                squeeze_dim = 1
            else:
                squeeze_dim = None
            input = sources
        else:
            raise ValueError("Only supports 3D or 4D input.")

        mask = self.compute_mask(input)
        
        output = mask * mixture

        if squeeze_dim is not None:
            output = output.squeeze(dim=squeeze_dim)
        
        output = self.istft(output)
        T_pad = output.size(-1)

        output = F.pad(output, (0, T - T_pad))

        return output
    
    def compute_mask(self, input):
        raise NotImplementedError("Implement compute_mask().")
    
    def stft(self, input):
        fft_size, hop_size = self.fft_size, self.hop_size
        window = self.window
        output = torch.stft(input, fft_size, hop_length=hop_size, window=window, return_complex=True)

        return output

    def istft(self, input):
        fft_size, hop_size = self.fft_size, self.hop_size
        window = self.window
        output = torch.istft(input, fft_size, hop_length=hop_size, window=window, return_complex=False)

        return output

class IdealBinaryMasking(FrequencyMasking):
    def __init__(self, fft_size, hop_size=None, window_fn='hann', domain='time'):
        super().__init__(fft_size, hop_size=hop_size, window_fn=window_fn, domain=domain)
    
    def compute_mask(self, input):
        mask = ideal_binary_mask(input)
        return mask

class IdealRatioMasking(FrequencyMasking):
    def __init__(self, fft_size, hop_size=None, window_fn='hann', domain='time'):
        super().__init__(fft_size, hop_size=hop_size, window_fn=window_fn, domain=domain)
    
    def compute_mask(self, input):
        mask = ideal_ratio_mask(input)
        return mask

class PhaseSensitiveMasking(FrequencyMasking):
    def __init__(self, fft_size, hop_size=None, window_fn='hann', domain='time'):
        super().__init__(fft_size, hop_size=hop_size, window_fn=window_fn, domain=domain)
    
    def compute_mask(self, input):
        mask = phase_sensitive_mask(input)
        return mask