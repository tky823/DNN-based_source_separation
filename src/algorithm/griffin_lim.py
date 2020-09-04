import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.stft import BatchSTFT, BatchInvSTFT

class GriffinLim(nn.Module):
    def __init__(self, fft_size, hop_size=None, window_fn='hann'):
        super().__init__()
        if hop_size is None:
            hop_size = fft_size//4
        
        self.fft_size = fft_size
        
        self.stft = BatchSTFT(fft_size, hop_size=hop_size, window_fn=window_fn, normalize=True)
        self.istft = BatchInvSTFT(fft_size, hop_size=hop_size, window_fn=window_fn, normalize=True)
    
    def forward(self, amplitude, phase=None, iteration=10):
        """
            Args:
                amplitude (F_bin, T_bin)
            Returns:
                phase (F_bin, T_bin): Reconstructed phase
        """
        for idx in range(iteration):
            phase = self.update(amplitude, phase)
        
        return phase
    
    def update(self, amplitude, phase=None):
        """
            Args:
                amplitude (F_bin, T_bin)
            Returns:
                phase (F_bin, T_bin)
        """
        F_bin, T_bin = amplitude.size()
        
        if phase is None:
            sampler = torch.distributions.uniform.Uniform(0, 2*math.pi)
            phase = sampler.sample((F_bin, T_bin)).to(amplitude.device)
        
        real, imag = amplitude * torch.cos(phase), amplitude * torch.sin(phase)
        spectrogram = torch.cat([real, imag], dim=0).unsqueeze(dim=0) # (1, 2*F_bin, T_bin)
        
        signal = self.istft(spectrogram)
        spectrogram = self.stft(signal)
        
        spectrogram = spectrogram.squeeze(dim=0)
        real, imag = spectrogram[:F_bin], spectrogram[F_bin:]
        phase = torch.atan2(imag, real)
        
        return phase
            
if __name__ == '__main__':
    import numpy as np
    from scipy.signal import resample_poly
    
    from utils.utils_audio import read_wav, write_wav
    
    torch.manual_seed(111)
    
    fft_size, hop_size = 1024, 256
    n_basis = 4
    
    signal, sr = read_wav("data/man-44100.wav")
    signal = resample_poly(signal, up=16000, down=sr)
    write_wav("data/man-16000.wav", signal=signal, sr=16000)
    
    T = len(signal)
    signal = torch.Tensor(signal).unsqueeze(dim=0)
    
    stft = BatchSTFT(fft_size=fft_size, hop_size=hop_size)
    istft = BatchInvSTFT(fft_size=fft_size, hop_size=hop_size)
    
    spectrogram = stft(signal)
    oracle_signal = istft(spectrogram, T=T)
    oracle_signal = oracle_signal.squeeze(dim=0).numpy()
    write_wav("data/man-oracle.wav", signal=oracle_signal, sr=16000)
    
    griffin_lim = GriffinLim(fft_size, hop_size=hop_size)
    
    spectrogram = spectrogram.squeeze(dim=0)
    real = spectrogram[:fft_size//2+1]
    imag = spectrogram[fft_size//2+1:]
    amplitude = torch.sqrt(real**2+imag**2)
    
    # Griffin-Lim iteration 10
    iteration = 10
    estimated_phase = griffin_lim(amplitude, iteration=iteration)
    
    real, imag = amplitude * torch.cos(estimated_phase), amplitude * torch.sin(estimated_phase)
    estimated_spectrogram = torch.cat([real, imag], dim=0) # (2*F_bin, T_bin)
    estimated_spectrogram = estimated_spectrogram.unsqueeze(dim=0)
    
    estimated_signal = istft(estimated_spectrogram, T=T)
    estimated_signal = estimated_signal.squeeze(dim=0).numpy()
    write_wav("data/man-estimated_GL{}.wav".format(iteration), signal=estimated_signal, sr=16000)
    
    # Griffin-Lim iteration 10
    iteration = 50
    estimated_phase = griffin_lim(amplitude, iteration=iteration)
    
    real, imag = amplitude * torch.cos(estimated_phase), amplitude * torch.sin(estimated_phase)
    estimated_spectrogram = torch.cat([real, imag], dim=0) # (2*F_bin, T_bin)
    estimated_spectrogram = estimated_spectrogram.unsqueeze(dim=0)
    
    estimated_signal = istft(estimated_spectrogram, T=T)
    estimated_signal = estimated_signal.squeeze(dim=0).numpy()
    write_wav("data/man-estimated_GL{}.wav".format(iteration), signal=estimated_signal, sr=16000)
    
