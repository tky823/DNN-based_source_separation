import numpy as np
import torch

def build_window(n_fft, window_fn='hann', **kwargs):
    if window_fn=='hann':
        assert set(kwargs) == set(), "kwargs is expected empty but given kwargs={}.".format(kwargs)
        window = torch.hann_window(n_fft, periodic=True)
    elif window_fn=='hamming':
        assert set(kwargs) == set(), "kwargs is expected empty but given kwargs={}.".format(kwargs)
        window = torch.hamming_window(n_fft, periodic=True)
    elif window_fn == 'blackman':
        window = torch.blackman_window(n_fft, periodic=True)
    elif window_fn=='kaiser':
        assert set(kwargs) == {'beta'}, "kwargs is expected to include key `beta` but given kwargs={}.".format(kwargs)
        window = torch.kaiser_window(n_fft, beta=kwargs['beta'], periodic=True)
    else:
        raise ValueError("Not support {} window.".format(window_fn))
    
    return window
    
def build_optimal_window(window, hop_length=None):
    """
    Args:
        window: (window_length,)
    """
    window_length = len(window)

    if hop_length is None:
        hop_length = window_length // 2

    windows = torch.cat([
        torch.roll(window.unsqueeze(dim=0), hop_length*idx) for idx in range(window_length // hop_length)
    ], dim=0)
    
    norm = torch.sum(windows**2, dim=0)
    optimal_window = window / norm
    
    return optimal_window

def load_piano_roll(midi_path, sample_rate, hop_length, dtype=torch.uint8):
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(midi_path)
    piano_roll_sample_rate = sample_rate / hop_length
    piano_roll = midi.get_piano_roll(fs=piano_roll_sample_rate)
    piano_roll = piano_roll.astype(np.uint8)
    piano_roll = torch.from_numpy(piano_roll)

    if dtype is torch.uint8:
        pass
    elif dtype in [torch.float, torch.float64]:
        piano_roll = piano_roll / 128
    else:
        raise ValueError("Invalid dtype is specified.")

    return piano_roll

def _test_piano_roll():
    n_fft, hop_length = 4096, 1024

    waveform, sample_rate = torchaudio.load("data/midi/CDEFGGGAFCAGCG.wav")
    window = torch.hann_window(n_fft)
    spectrogram = torch.stft(waveform, n_fft, hop_length=hop_length, window=window, center=True, onesided=True, return_complex=True)
    log_spectrogram = 20 * torch.log10(torch.abs(spectrogram) + 1e-12)
    piano_roll = load_piano_roll("data/midi/CDEFGGGAFCAGCG.mid", sample_rate=sample_rate, hop_length=hop_length, dtype=torch.float)
    
    print(piano_roll.size())

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(log_spectrogram[0, :500], cmap="jet")
    plt.savefig("data/midi/CDEFGGGAFCAGCG-spectrogram.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(piano_roll, cmap="binary")
    plt.grid()
    plt.savefig("data/midi/CDEFGGGAFCAGCG-midi.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torchaudio

    plt.rcParams["font.size"] = 18

    _test_piano_roll()