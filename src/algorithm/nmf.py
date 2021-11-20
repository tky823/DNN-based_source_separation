import torch

from criterion.divergence import generalized_kl_divergence, is_divergence

EPS=1e-12

__metrics__ = ['EUC', 'KL', 'IS']

class NMF:
    def __init__(self, K=2, metric='EUC', eps=EPS):
        """
        Args:
            K: number of bases
            metric: 'EUC', 'KL', 'IS'
        """
        assert metric in __metrics__, "metric is expected any of {}, given {}".format(metric, __metrics__)

        self.K = K
        self.metric = metric
        self.eps = eps

        if metric == 'EUC':
            self.criterion = lambda input, target: (input - target)**2
        elif metric == 'KL':
            self.criterion = generalized_kl_divergence
        elif metric == 'IS':
            self.criterion = is_divergence
        
        self.loss = []
    
    def update(self, target, iteration=100):
        K = self.K
        self.target = target
        F_bin, T_bin = target.size()

        self.base = torch.rand(F_bin, K, dtype=torch.float) + 1
        self.activation = torch.rand(K, T_bin, dtype=torch.float) + 1
        self.reconstruction = torch.matmul(self.base, self.activation)

        loss = self.criterion(self.reconstruction, target)
        self.loss.append(loss.sum().item())

        for idx in range(iteration):
            self.update_once()
            loss = self.criterion(self.reconstruction, target)
            self.loss.append(loss.sum().item())
        
    def update_once(self):
        if self.metric == 'EUC':
            self.update_euc()
        elif self.metric == 'KL':
            self.update_kl()
        elif self.metric == 'IS':
            self.update_is()
        else:
            raise NotImplementedError("Not support {}".format(self.metric))

        self.reconstruction = torch.matmul(self.base, self.activation)

    def update_euc(self):
        eps = self.eps
        target = self.target
        base, activation, reconstruction = self.base, self.activation, self.reconstruction
        base_transpose, activation_transpose = base.permute(1,0), activation.permute(1,0)

        self.base = base * (torch.matmul(target, activation_transpose) / (torch.matmul(reconstruction, activation_transpose) + eps))
        self.activation = activation * (torch.matmul(base_transpose, target) / (torch.matmul(base_transpose, reconstruction) + eps))
    
    def update_kl(self):
        eps = self.eps
        target = self.target
        base, activation, reconstruction = self.base, self.activation, self.reconstruction
        base_transpose, activation_transpose = base.permute(1,0), activation.permute(1,0)
        division = target / (reconstruction + eps)
        
        self.base = base * (torch.matmul(division, activation_transpose) / (activation_transpose.sum(dim=0, keepdim=True) + eps))
        self.activation = activation * (torch.matmul(base_transpose, division) / (base_transpose.sum(dim=1, keepdim=True) + eps))
    
    def update_is(self):
        eps = self.eps
        target = self.target
        base, activation, reconstruction = self.base, self.activation, self.reconstruction
        base_transpose, activation_transpose = base.permute(1,0), activation.permute(1,0)
        division, reconstruction_inverse = target / (reconstruction + eps)**2, 1 / (reconstruction + eps)
        
        self.base = base * torch.sqrt(torch.matmul(division, activation_transpose) / (torch.matmul(reconstruction_inverse, activation_transpose) + eps))
        self.activation = activation * torch.sqrt(torch.matmul(base_transpose, division) / (torch.matmul(base_transpose, reconstruction_inverse) + eps))

def _test(metric='EUC'):
    torch.manual_seed(111)

    n_fft, hop_length = 1024, 256
    n_basis = 6
    iteration = 100
    
    signal, sr = read_wav("data/music-8000.wav")
    
    T = len(signal)
    signal = torch.Tensor(signal).unsqueeze(dim=0)
    
    stft = BatchSTFT(n_fft=n_fft, hop_length=hop_length)
    istft = BatchInvSTFT(n_fft=n_fft, hop_length=hop_length)

    spectrogram = stft(signal).squeeze(dim=0)
    real = spectrogram[...,0]
    imag = spectrogram[...,1]
    amplitude = torch.sqrt(real**2 + imag**2)
    power = amplitude**2

    log_spectrogram = 10 * torch.log10(power + EPS)
    plt.figure()
    plt.pcolormesh(log_spectrogram, cmap='jet')
    plt.colorbar()
    plt.savefig('data/NMF/spectrogram.png', bbox_inches='tight')
    plt.close()

    nmf = NMF(n_basis, metric=metric)
    nmf.update(power, iteration=iteration)

    estimated_power = torch.matmul(nmf.base, nmf.activation)
    estimated_amplitude = torch.sqrt(estimated_power)
    ratio = estimated_amplitude / (amplitude + EPS)
    estimated_real, estimated_imag = ratio * real, ratio * imag
    estimated_spectrogram = torch.cat([estimated_real.unsqueeze(dim=2), estimated_imag.unsqueeze(dim=2)], dim=2).unsqueeze(dim=0)

    estimated_signal = istft(estimated_spectrogram, T=T)
    estimated_signal = estimated_signal.squeeze(dim=0).numpy()
    estimated_signal = estimated_signal / np.abs(estimated_signal).max()
    write_wav("data/NMF/{}/music-8000-estimated-iter{}.wav".format(metric, iteration), signal=estimated_signal, sr=8000)

    for idx in range(n_basis):
        estimated_power = torch.matmul(nmf.base[:, idx: idx+1], nmf.activation[idx: idx+1, :])
        estimated_amplitude = torch.sqrt(estimated_power)
        ratio = estimated_amplitude / (amplitude + EPS)
        estimated_real, estimated_imag = ratio * real, ratio * imag
        estimated_spectrogram = torch.cat([estimated_real.unsqueeze(dim=2), estimated_imag.unsqueeze(dim=2)], dim=2).unsqueeze(dim=0)

        estimated_signal = istft(estimated_spectrogram, T=T)
        estimated_signal = estimated_signal.squeeze(dim=0).numpy()
        estimated_signal = estimated_signal / np.abs(estimated_signal).max()
        write_wav("data/NMF/{}/music-8000-estimated-iter{}-base{}.wav".format(metric, iteration, idx), signal=estimated_signal, sr=8000)

        log_spectrogram = 10 * torch.log10(estimated_power + EPS).numpy()
        plt.figure()
        plt.pcolormesh(log_spectrogram, cmap='jet')
        plt.colorbar()
        plt.savefig('data/NMF/{}/estimated-spectrogram-iter{}-base{}.png'.format(metric, iteration, idx), bbox_inches='tight')
        plt.close()
    
    plt.figure()
    plt.plot(nmf.loss)
    plt.savefig('data/NMF/{}/loss.png'.format(metric), bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    
    from utils.utils_audio import read_wav, write_wav
    from algorithm.stft import BatchSTFT, BatchInvSTFT

    os.makedirs('data/NMF/EUC', exist_ok=True)
    os.makedirs('data/NMF/KL', exist_ok=True)
    os.makedirs('data/NMF/IS', exist_ok=True)

    _test(metric='EUC')
    _test(metric='KL')
    _test(metric='IS')
