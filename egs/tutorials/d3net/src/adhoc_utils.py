import os
import math

import torch
import torchaudio
import torch.nn.functional as F

from models.d3net import D3Net, ParallelD3Net

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

def separate_by_d3net(filepath, out_dir):
    patch_size = 256
    fft_size, hop_size = 4096, 1024
    window = torch.hann_window(fft_size)

    x, sample_rate = torchaudio.load(filepath)
    _, T = x.size()
    model = load_d3net()

    assert sample_rate == SAMPLE_RATE_MUSDB18, "sample rate must be {}, but given {}".format(SAMPLE_RATE_MUSDB18, sample_rate)

    mixture = torch.stft(x, n_fft=fft_size, hop_length=hop_size, window=window, return_complex=True)
    padding = (patch_size - mixture.size(-1) % patch_size) % patch_size

    mixture = F.pad(mixture, (0, padding))
    mixture = mixture.reshape(*mixture.size()[:2], -1, patch_size)
    mixture = mixture.permute(2, 0, 1, 3).unsqueeze(dim=1)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        mixture = mixture.cuda()

    n_sources = len(__sources__)

    model.eval()

    with torch.no_grad():
        batch_size, _, n_mics, n_bins, n_frames = mixture.size()
        
        mixture_amplitude = torch.abs(mixture)
        
        estimated_sources_amplitude = {
            target: [] for target in __sources__
        }

        # Serial operation
        for _mixture_amplitude in mixture_amplitude:
            # _mixture_amplitude: (1, n_mics, n_bins, n_frames)
            for target in __sources__:
                _estimated_sources_amplitude = model(_mixture_amplitude, target=target)
                estimated_sources_amplitude[target].append(_estimated_sources_amplitude)
        
        estimated_sources_amplitude = [
            torch.cat(estimated_sources_amplitude[target], dim=0).unsqueeze(dim=0) for target in __sources__
        ]
        estimated_sources_amplitude = torch.cat(estimated_sources_amplitude, dim=0) # (n_sources, batch_size, n_mics, n_bins, n_frames)
        estimated_sources_amplitude = estimated_sources_amplitude.permute(0, 2, 3, 1, 4)
        estimated_sources_amplitude = estimated_sources_amplitude.reshape(n_sources, n_mics, n_bins, batch_size * n_frames) # (n_sources, n_mics, n_bins, T_pad)

        mixture = mixture.permute(1, 2, 3, 0, 4).reshape(1, n_mics, n_bins, batch_size * n_frames) # (1, n_mics, n_bins, T_pad)

        mixture = mixture.cpu()
        estimated_sources_amplitude = estimated_sources_amplitude.cpu()

        estimated_sources = apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude=estimated_sources_amplitude)
        estimated_sources_channels = estimated_sources.size()[:-2]

        estimated_sources = estimated_sources.view(-1, *estimated_sources.size()[-2:])
        estimated_sources = torch.istft(estimated_sources, fft_size, hop_length=hop_size, window=window, return_complex=False)
        estimated_sources = estimated_sources.view(*estimated_sources_channels, -1) # -> (n_sources, n_mics, T_pad)

        os.makedirs(out_dir, exist_ok=True)
        estimated_paths = {}

        for idx in range(n_sources):
            source = __sources__[idx]
            path = os.path.join(out_dir, "{}.wav".format(source))
            torchaudio.save(path, estimated_sources[idx][:, :T], sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
            estimated_paths[source] = path
        
        return estimated_paths

def load_d3net():
    modules = {}
    model_dir = "./model"

    for source in __sources__:
        model_path = os.path.join(model_dir, source, "last.pth")
        modules[source] = D3Net.build_model(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        modules[source].load_state_dict(package['state_dict'])

    model = ParallelD3Net(modules)

    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        print("Uses CUDA")
    else:
        print("Does NOT use CUDA")
    
    return model

def apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude, iteration=1, channels_first=True, eps=EPS):
    """
    Multichannel Wiener filter.
    Implementation is based on norbert package
    """
    assert channels_first, "`channels_first` is expected True, but given {}".format(channels_first)

    n_dims = mixture.dim()

    if n_dims == 4:
        mixture = mixture.squeeze(dim=0)
    elif n_dims != 3:
        raise ValueError("mixture.dim() is expected 3 or 4, but given {}.".format(mixture.dim()))

    assert estimated_sources_amplitude.dim() == 4, "estimated_sources_amplitude.dim() is expected 4, but given {}.".format(estimated_sources_amplitude.dim())

    # Use soft mask
    ratio = estimated_sources_amplitude / (estimated_sources_amplitude.sum(dim=0) + eps)
    estimated_sources = ratio * mixture

    norm = max(1, torch.abs(mixture).max() / 10)
    mixture, estimated_sources = mixture / norm, estimated_sources / norm

    estimated_sources = update_em(mixture, estimated_sources, iteration, eps=eps)
    estimated_sources = norm * estimated_sources

    return estimated_sources

def update_em(mixture, estimated_sources, iterations=1, source_parallel=False, eps=EPS):
    """
    Args:
        mixture: (n_channels, n_bins, n_frames)
        estimated_sources: (n_sources, n_channels, n_bins, n_frames)
    Returns
        estiamted_sources: (n_sources, n_channels, n_bins, n_frames)
    """
    n_sources, n_channels, _, _ = estimated_sources.size()

    for iteration_idx in range(iterations):
        v, R = [], []
        Cxx = 0

        if source_parallel:
            v, R = get_stats(estimated_sources, eps=eps) # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_channels, n_channels)
            Cxx = torch.sum(v.unsqueeze(dim=4) * R, dim=0) # (n_bins, n_frames, n_channels, n_channels)
        else:
            for source_idx in range(n_sources):
                y_n = estimated_sources[source_idx] # (n_channels, n_bins, n_frames)
                v_n, R_n = get_stats(y_n, eps=eps) # (n_bins, n_frames), (n_bins, n_channels, n_channels)
                Cxx = Cxx + v_n.unsqueeze(dim=2).unsqueeze(dim=3) * R_n.unsqueeze(dim=1) # (n_bins, n_frames, n_channels, n_channels)
                v.append(v_n.unsqueeze(dim=0))
                R.append(R_n.unsqueeze(dim=0))
        
            v, R = torch.cat(v, dim=0), torch.cat(R, dim=0) # (n_sources, n_bins, n_frames), (n_sources, n_bins, n_channels, n_channels)
       
        v, R = v.unsqueeze(dim=3), R.unsqueeze(dim=2) # (n_sources, n_bins, n_frames, 1), (n_sources, n_bins, 1, n_channels, n_channels)

        inv_Cxx = torch.linalg.inv(Cxx + math.sqrt(eps) * torch.eye(n_channels)) # (n_bins, n_frames, n_channels, n_channels)

        if source_parallel:
            gain = v.unsqueeze(dim=4) * torch.sum(R.unsqueeze(dim=5) * inv_Cxx.unsqueeze(dim=2), dim=4) # (n_sources, n_bins, n_frames, n_channels, n_channels)
            gain = gain.permute(0, 3, 4, 1, 2) # (n_sources, n_channels, n_channels, n_bins, n_frames)
            estimated_sources = torch.sum(gain * mixture, dim=2) # (n_sources, n_channels, n_bins, n_frames)
        else:
            estimated_sources = []

            for source_idx in range(n_sources):
                v_n, R_n = v[source_idx], R[source_idx] # (n_bins, n_frames, 1), (n_bins, 1, n_channels, n_channels)

                gain_n = v_n.unsqueeze(dim=3) * torch.sum(R_n.unsqueeze(dim=4) * inv_Cxx.unsqueeze(dim=2), dim=3) # (n_bins, n_frames, n_channels, n_channels)
                gain_n = gain_n.permute(2, 3, 0, 1) # (n_channels, n_channels, n_bins, n_frames)
                estimated_source = torch.sum(gain_n * mixture, dim=1) # (n_channels, n_bins, n_frames)
                estimated_sources.append(estimated_source.unsqueeze(dim=0))
            
            estimated_sources = torch.cat(estimated_sources, dim=0) # (n_sources, n_channels, n_bins, n_frames)

    return estimated_sources

def get_stats(spectrogram, eps=EPS):
    """
    Compute empirical parameters of local gaussian model.
    Args:
        spectrogram <torch.Tensor>: (n_mics, n_bins, n_frames) or (n_sources, n_mics, n_bins, n_frames)
    Returns:
        psd <torch.Tensor>: (n_bins, n_frames) or (n_sources, n_bins, n_frames)
        covariance <torch.Tensor>: (n_bins, n_frames, n_mics, n_mics) or (n_sources, n_bins, n_frames, n_mics, n_mics)
    """
    n_dims = spectrogram.dim()

    if n_dims == 3:
        psd = torch.mean(torch.abs(spectrogram)**2, dim=0) # (n_bins, n_frames)
        covariance = spectrogram.unsqueeze(dim=1) * spectrogram.unsqueeze(dim=0).conj() # (n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.sum(dim=3) # (n_mics, n_mics, n_bins)
        denominator = psd.sum(dim=1) + eps # (n_bins,)

        covariance = covariance / denominator # (n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.permute(2, 0, 1) # (n_bins, n_mics, n_mics)
    elif n_dims == 4:
        psd = torch.mean(torch.abs(spectrogram)**2, dim=1) # (n_sources, n_bins, n_frames)
        covariance = spectrogram.unsqueeze(dim=2) * spectrogram.unsqueeze(dim=1).conj() # (n_sources, n_mics, n_mics, n_bins, n_frames)
        covariance = covariance.sum(dim=4) # (n_sources, n_mics, n_mics, n_bins)
        denominator = psd.sum(dim=2) + eps # (n_sources, n_bins)
        
        covariance = covariance / denominator.unsqueeze(dim=1).unsqueeze(dim=2) # (n_sources, n_mics, n_mics, n_bins)
        covariance = covariance.permute(0, 3, 1, 2) # (n_sources, n_bins, n_mics, n_mics)
    else:
        raise ValueError("Invalid dimension of tensor is given.")

    return psd, covariance