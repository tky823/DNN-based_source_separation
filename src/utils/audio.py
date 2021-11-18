import torch

def build_window(fft_size, window_fn='hann', **kwargs):
    if window_fn=='hann':
        assert set(kwargs) == set(), "kwargs is expected empty but given kwargs={}.".format(kwargs)
        window = torch.hann_window(fft_size, periodic=True)
    elif window_fn=='hamming':
        assert set(kwargs) == set(), "kwargs is expected empty but given kwargs={}.".format(kwargs)
        window = torch.hamming_window(fft_size, periodic=True)
    elif window_fn == 'blackman':
        window = torch.blackman_window(fft_size, periodic=True)
    elif window_fn=='kaiser':
        assert set(kwargs) == {'beta'}, "kwargs is expected to include key `beta` but given kwargs={}.".format(kwargs)
        window = torch.kaiser_window(fft_size, beta=kwargs['beta'], periodic=True)
    else:
        raise ValueError("Not support {} window.".format(window_fn))
    
    return window
    
def build_optimal_window(window, hop_size=None):
    """
    Args:
        window: (window_length,)
    """
    window_length = len(window)

    if hop_size is None:
        hop_size = window_length // 2

    windows = torch.cat([
        torch.roll(window.unsqueeze(dim=0), hop_size*idx) for idx in range(window_length // hop_size)
    ], dim=0)
    
    norm = torch.sum(windows**2, dim=0)
    optimal_window = window / norm
    
    return optimal_window