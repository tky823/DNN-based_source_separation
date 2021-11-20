import torch

EPS = 1e-12

def real_cepstrum(input, n_fft=None, minimum_distortion=False, eps=EPS):
    if torch.is_complex(input):
        raise ValueError("input should be real.")
    
    if n_fft is None:
        n_fft = input.size(-1)
    
    cepstrum = torch.fft.irfft(torch.log(torch.abs(torch.fft.rfft(input, n_fft)) + eps), n_fft)
    
    if minimum_distortion:
        odd = n_fft % 2
        ones_left, ones_center, ones_right = torch.ones(1), torch.ones((n_fft + odd) // 2 - 1), torch.ones(1 - odd)
        zeros = torch.zeros((n_fft + odd) // 2 - 1)
        window = torch.cat([ones_left, 2 * ones_center, ones_right, zeros])
        window = window.to(cepstrum.device)
        cepstrum = torch.fft.irfft(torch.exp(torch.fft.rfft(window * cepstrum, n_fft)), n_fft)
    
    return cepstrum

def _test_rceps():
    waveform, _ = torchaudio.load("./data/single-channel/mtlb.wav")
    cepstrum = real_cepstrum(waveform)
    print(cepstrum)

    minimum_distortion_cepstrum = real_cepstrum(waveform, onesided=True)
    print(minimum_distortion_cepstrum)

def _test_rceps_echo_cancel():
    waveform, sample_rate = torchaudio.load("./data/single-channel/mtlb.wav")

    lag, alpha = 0.23, 0.5
    delta = round(lag * sample_rate)

    silence = torch.zeros(waveform.size(0), delta)
    orig = torch.cat([waveform, silence], dim=1)
    echo = alpha * torch.cat([silence, waveform], dim=1)
    reverbed = orig + echo

    cepstrum = real_cepstrum(reverbed)
    print(cepstrum)

    plt.figure()
    plt.plot(cepstrum.squeeze())
    plt.show()
    plt.close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torchaudio

    _test_rceps()
    print()

    # _test_rceps_echo_cancel()