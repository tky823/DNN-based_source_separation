import os

import norbert
import torch
import torchaudio
import torch.nn.functional as F

from models.d3net import D3Net, ParallelD3Net

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
BITS_PER_SAMPLE_MUSDB18 = 16

def separate_by_d3net(filepath):
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

        estimated_sources = apply_multichannel_wiener_filter(mixture, estimated_sources_amplitude)
        estimated_sources_channels = estimated_sources.size()[:-2]

        estimated_sources = estimated_sources.view(-1, *estimated_sources.size()[-2:])
        estimated_sources = torch.istft(estimated_sources, fft_size, hop_length=hop_size, window=window, return_complex=False)
        estimated_sources = estimated_sources.view(*estimated_sources_channels, -1) # -> (n_sources, n_mics, T_pad)

        os.makedirs("./estimations", exist_ok=True)
        estimations = {}

        for idx in range(n_sources):
            source = __sources__[idx]
            path = "./estimations/{}.wav".format(source)
            torchaudio.save(path, estimated_sources[idx][:, :T], sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
            estimations[source] = path
        
        return estimations

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

def apply_multichannel_wiener_filter(mixture, estimated_amplitude, channels_first=True, eps=1e-12):
    """
    Args:
        mixture <torch.Tensor>: (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames), complex tensor
        estimated_amplitude <torch.Tensor>: (n_sources, n_channels, n_bins, n_frames), real (nonnegative) tensor
    Returns:
        estimated_sources <torch.Tensor>: (n_sources, n_channels, n_bins, n_frames), complex tensor
    """
    assert channels_first, "`channels_first` is expected True, but given {}".format(channels_first)

    n_dims = mixture.dim()

    if n_dims == 4:
        mixture = mixture.squeeze(dim=0)
    elif n_dims != 3:
        raise ValueError("mixture.dim() is expected 3 or 4, but given {}.".format(mixture.dim()))

    assert estimated_amplitude.dim() == 4, "estimated_amplitude.dim() is expected 4, but given {}.".format(estimated_amplitude.dim())

    device = mixture.device
    dtype = mixture.dtype

    mixture = mixture.detach().cpu().numpy()
    estimated_amplitude = estimated_amplitude.detach().cpu().numpy()

    mixture = mixture.transpose(2, 1, 0)
    estimated_amplitude = estimated_amplitude.transpose(3, 2, 1, 0)
    estimated_sources = norbert.wiener(estimated_amplitude, mixture, iterations=1, eps=eps)
    estimated_sources = estimated_sources.transpose(3, 2, 1, 0)
    estimated_sources = torch.from_numpy(estimated_sources).to(device, dtype)

    return estimated_sources