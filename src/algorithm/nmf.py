import torch
import torch.nn as nn

EPS=1e-12

__metrics__ = ['EU', 'KL', 'IS']

class NMF(nn.Module):
    def __init__(self, K=2, metric='EU'):
        """
        Args:
            K: number of bases
            metric: 'EU', 'KL', 'IS'
        """
        super().__init__()
        
        assert metric in __metrics__, "metric is expected any of {}, given {}".format(metric, __metrics__)

        self.K = K
        self.metric = metric
    
    def forward(self, input, iteration=10):
        """
        Args:
            input (F_bin, T_bin)
            iteration : iterations of update
        """
        K = self.K
        self.x = input
        F_bin, T_bin = input.size()

        self.basis = torch.rand(F_bin, K, dtype=torch.float) + 1
        self.activation = torch.rand(K, T_bin, dtype=torch.float) + 1

        for idx in range(iteration):
            self.update()
    
    def update(self):
        if self.metric == 'EU':
            self.update_euc()
        else:
            raise NotImplementedError("Not support {}".format(self.metric))

    def update_euc(self):
        x = self.x
        basis, activation = self.basis, self.activation
        basis_transpose, activation_transpose = basis.permute(1,0), activation.permute(1,0)
        
        basis_next =  basis * torch.matmul(x, activation_transpose) / torch.matmul(basis, torch.matmul(activation, activation_transpose))
        activation_next = activation * torch.matmul(basis_transpose, x) / torch.matmul(torch.matmul(basis_transpose, basis), activation)     
        self.basis = torch.where(torch.isnan(basis_next), torch.zeros_like(basis_next), basis_next)
        self.activation = torch.where(torch.isnan(activation_next), torch.zeros_like(activation_next), activation_next)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import BatchSTFT, BatchInvSTFT

    torch.manual_seed(111)

    nmf = NMF()    
    
    fft_size, hop_size = 1024, 256
    n_basis = 8
    iteration = 100
    
    signal, sr = read_wav("data/music-8000.wav")
    
    T = len(signal)
    signal = torch.Tensor(signal).unsqueeze(dim=0)
    
    stft = BatchSTFT(fft_size=fft_size, hop_size=hop_size)
    istft = BatchInvSTFT(fft_size=fft_size, hop_size=hop_size)

    spectrogram = stft(signal).squeeze(dim=0)
    real = spectrogram[:fft_size//2+1]
    imag = spectrogram[fft_size//2+1:]
    amplitude = torch.sqrt(real**2+imag**2)
    power = amplitude**2

    log_spectrogram = 10*torch.log10(power + EPS)
    plt.figure()
    plt.pcolormesh(log_spectrogram, cmap='jet')
    plt.colorbar()
    plt.savefig('data/spectrogram.png', bbox_inches='tight')
    plt.close()
    
    nmf = NMF(n_basis)
    nmf(power)

    estimated_power = torch.matmul(nmf.basis, nmf.activation)
    estimated_amplitude = torch.sqrt(estimated_power)
    ratio = estimated_amplitude / (amplitude + EPS)
    real, imag = ratio * real, ratio * imag
    estimated_spectrogram = torch.cat([real, imag], dim=0)
    estimated_spectrogram = estimated_spectrogram.unsqueeze(dim=0)

    estimated_signal = istft(estimated_spectrogram, T=T)
    estimated_signal = estimated_signal.squeeze(dim=0).numpy()
    estimated_signal = estimated_signal / np.abs(estimated_signal).max()
    write_wav("data/music-8000-estimated_NMF-EU{}.wav".format(iteration), signal=estimated_signal, sr=8000)

    for idx in range(n_basis):
        estimated_power = torch.matmul(nmf.basis[:, idx: idx+1], nmf.activation[idx: idx+1, :])
        log_spectrogram = 10*torch.log10(estimated_power + EPS).numpy()
        plt.figure()
        plt.pcolormesh(log_spectrogram, cmap='jet')
        plt.colorbar()
        plt.savefig('data/estimated-spectrogram-{}.png'.format(idx), bbox_inches='tight')
        plt.close()
