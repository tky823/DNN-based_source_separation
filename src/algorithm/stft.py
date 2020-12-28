import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_audio import build_Fourier_bases, build_window, build_optimal_window

class BatchSTFT(nn.Module):
    def __init__(self, fft_size, hop_size=None, window_fn='hann', normalize=False):
        super().__init__()
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size
    
        window = build_window(fft_size, window_fn=window_fn) # (fft_size,)

        cos_bases, sin_bases = build_Fourier_bases(fft_size, normalize=normalize)
        cos_bases, sin_bases = cos_bases[:fft_size//2+1] * window, - sin_bases[:fft_size//2+1] * window
        
        bases = torch.cat([cos_bases, sin_bases], dim=0)
        
        self.bases = nn.Parameter(bases.unsqueeze(dim=1), requires_grad=False)
        
    def forward(self, input):
        """
        Args:
            input (batch_size, T)
        Returns:
            output (batch_size, n_bins, n_frames, 2): n_bins = fft_size//2+1, n_frames = (T - fft_size)//hop_size + 1. n_frames may be different because of padding.
        """
        batch_size, T = input.size()
    
        fft_size, hop_size = self.fft_size, self.hop_size
        n_bins = fft_size//2 + 1
        
        padding = (hop_size - (T - fft_size)%hop_size)%hop_size + 2 * fft_size # Assume that "fft_size%hop_size is 0"
        padding_left = padding // 2
        padding_right = padding - padding_left
        
        input = F.pad(input, (padding_left, padding_right))
        input = input.unsqueeze(dim=1)
        output = F.conv1d(input, self.bases, stride=self.hop_size)
        real, imag = output[:, :n_bins], output[:, n_bins:]
        output = torch.cat([real.unsqueeze(dim=3), imag.unsqueeze(dim=3)], dim=3)
        
        return output

class BatchInvSTFT(nn.Module):
    def __init__(self, fft_size, hop_size=None, window_fn='hann', normalize=False):
        super().__init__()
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size

        window = build_window(fft_size, window_fn=window_fn) # (fft_size,)
        optimal_window = build_optimal_window(window, hop_size=hop_size)

        cos_bases, sin_bases = build_Fourier_bases(fft_size, normalize=normalize)
        cos_bases, sin_bases = cos_bases[:fft_size//2+1] * optimal_window, - sin_bases[:fft_size//2+1] * optimal_window
        
        if not normalize:
            cos_bases = cos_bases / fft_size
            sin_bases = sin_bases / fft_size
        
        bases = torch.cat([cos_bases, sin_bases], dim=0)
        
        self.bases = nn.Parameter(bases.unsqueeze(dim=1), requires_grad=False)
        
    def forward(self, input, T=None):
        """
        Args:
            input (batch_size, n_bins, n_frames, 2): n_bins = fft_size//2+1, n_frames = (T - fft_size)//hop_size + 1. n_frames may be different because of padding.
        Returns:
            output (batch_size, T):
        """
        fft_size, hop_size = self.fft_size, self.hop_size
        
        if T is None:
            padding = 2 * fft_size
        else:
            padding = (hop_size - (T - fft_size)%hop_size)%hop_size + 2 * fft_size # Assume that "fft_size%hop_size is 0"
        padding_left = padding // 2
        padding_right = padding - padding_left
        
        real, imag = input[...,0], input[...,1]
        input = torch.cat([real, imag, real[:,1:-1], imag[:,1:-1]], dim=1)
        bases = torch.cat([self.bases, self.bases[1:fft_size//2], self.bases[-fft_size//2:-1]], dim=0)
        
        output = F.conv_transpose1d(input, bases, stride=self.hop_size)
        output = F.pad(output, (-padding_left, -padding_right))
        output = output.squeeze(dim=1)
        
        return output

if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    os.makedirs("data/STFT", exist_ok=True)

    torch.manual_seed(111)

    batch_size = 2
    T = 64
    fft_size, hop_size = 8, 2
    window_fn = 'hamming'

    input = torch.randn((batch_size, T), dtype=torch.float)
    
    stft = BatchSTFT(fft_size=fft_size, hop_size=hop_size, window_fn=window_fn)
    istft = BatchInvSTFT(fft_size=fft_size, hop_size=hop_size, window_fn=window_fn)
    spectrogram = stft(input)
    
    real, imag = spectrogram[...,0], spectrogram[...,1]
    power = real**2+imag**2

    plt.figure()
    plt.pcolormesh(real[0], cmap='bwr')
    plt.colorbar()
    plt.savefig('data/STFT/spectrogram_real.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.pcolormesh(imag[0], cmap='bwr')
    plt.colorbar()
    plt.savefig('data/STFT/spectrogram_imag.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.pcolormesh(power[0], cmap='bwr')
    plt.colorbar()
    plt.savefig('data/STFT/power.png', bbox_inches='tight')
    plt.close()
    
    output = istft(spectrogram, T=T)
    print(input.size(), output.size())

    plt.figure()
    plt.plot(range(T), input[0].numpy())
    plt.plot(range(T), output[0].numpy())
    plt.savefig('data/STFT/Fourier.png', bbox_inches='tight')
    plt.close()
