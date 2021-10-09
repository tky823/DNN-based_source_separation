import os

import torch
import torchaudio
import torch.nn.functional as F

from algorithm.frequency_mask import ideal_ratio_mask, multichannel_wiener_filter
from models.umx import OpenUnmix, ParallelOpenUnmix

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
NUM_CHANNELS_MUSDB18 = 2
BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

def separate_by_umx(model_paths, file_paths, out_dirs):
    patch_size = 256
    fft_size, hop_size = 4096, 1024
    window = torch.hann_window(fft_size)

    use_cuda = torch.cuda.is_available()

    model = load_pretrained_umx(model_paths)
    
    if use_cuda:
        model.cuda()
        print("Uses CUDA")
    else:
        print("Does NOT use CUDA")

    model.eval()

    estimated_paths = []

    for file_path, out_dir in zip(file_paths, out_dirs):
        x, sample_rate = torchaudio.load(file_path)
        _, T_original = x.size()

        if sample_rate == SAMPLE_RATE_MUSDB18:
            pre_resampler, post_resampler = None, None
        else:
            pre_resampler, post_resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE_MUSDB18), torchaudio.transforms.Resample(SAMPLE_RATE_MUSDB18, sample_rate)

        if pre_resampler is not None:
            x = pre_resampler(x)
        
        mixture = torch.stft(x, n_fft=fft_size, hop_length=hop_size, window=window, return_complex=True)
        padding = (patch_size - mixture.size(-1) % patch_size) % patch_size

        mixture = F.pad(mixture, (0, padding))
        mixture = mixture.reshape(*mixture.size()[:2], -1, patch_size)
        mixture = mixture.permute(2, 0, 1, 3).unsqueeze(dim=1)

        if use_cuda:
            mixture = mixture.cuda()

        n_sources = len(__sources__)

        with torch.no_grad():
            batch_size, _, n_mics, n_bins, n_frames = mixture.size()
            
            mixture_amplitude = torch.abs(mixture)
            
            estimated_sources_amplitude = {
                target: [] for target in __sources__
            }

            # Serial operation
            for _mixture_amplitude in mixture_amplitude:
                # _mixture_amplitude: (1, n_mics, n_bins, n_frames)
                if n_mics == 1:
                    _mixture_amplitude = torch.tile(_mixture_amplitude, (1, NUM_CHANNELS_MUSDB18, 1, 1))
                
                for target in __sources__:
                    _estimated_sources_amplitude = model(_mixture_amplitude, target=target)

                    if n_mics == 1:
                        _estimated_sources_amplitude = _estimated_sources_amplitude.mean(dim=1, keepdim=True)
                    
                    estimated_sources_amplitude[target].append(_estimated_sources_amplitude)
        
            estimated_sources_amplitude = [
                torch.cat(estimated_sources_amplitude[target], dim=0).unsqueeze(dim=0) for target in __sources__
            ]
            estimated_sources_amplitude = torch.cat(estimated_sources_amplitude, dim=0) # (n_sources, batch_size, n_mics, n_bins, n_frames)
            estimated_sources_amplitude = estimated_sources_amplitude.permute(0, 2, 3, 1, 4)
            estimated_sources_amplitude = estimated_sources_amplitude.reshape(n_sources, n_mics, n_bins, batch_size * n_frames) # (n_sources, n_mics, n_bins, batch_size * n_frames)

            mixture = mixture.permute(1, 2, 3, 0, 4).reshape(1, n_mics, n_bins, batch_size * n_frames) # (1, n_mics, n_bins, batch_size * n_frames)

            mixture = mixture.cpu()
            estimated_sources_amplitude = estimated_sources_amplitude.cpu()

            if n_mics == 1:
                estimated_sources_amplitude = estimated_sources_amplitude.squeeze(dim=1) # (n_sources, n_bins, batch_size * n_frames)
                mask = ideal_ratio_mask(estimated_sources_amplitude)
                mask = mask.unsqueeze(dim=1) # (n_sources, n_mics, n_bins, batch_size * n_frames)
                estimated_sources = mask * mixture # (n_sources, n_mics, n_bins, batch_size * n_frames)
            else:
                estimated_sources = apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude=estimated_sources_amplitude)
            
            estimated_sources_channels = estimated_sources.size()[:-2]

            estimated_sources = estimated_sources.view(-1, *estimated_sources.size()[-2:])
            y = torch.istft(estimated_sources, fft_size, hop_length=hop_size, window=window, return_complex=False)
            
            if post_resampler is not None:
                y = post_resampler(y)
            
            y = y.view(*estimated_sources_channels, -1) # -> (n_sources, n_mics, T_pad)
            T_pad = y.size(-1)
            y = F.pad(y, (0, T_original - T_pad)) # -> (n_sources, n_mics, T_original)

            os.makedirs(out_dir, exist_ok=True)
            _estimated_paths = {}

            for idx in range(n_sources):
                source = __sources__[idx]
                path = os.path.join(out_dir, "{}.wav".format(source))
                torchaudio.save(path, y[idx], sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                _estimated_paths[source] = path
            
            estimated_paths.append(_estimated_paths)
            
    return estimated_paths

def load_pretrained_umx(model_paths):
    modules = {}

    for source in __sources__:
        model_path = model_paths[source]
        modules[source] = OpenUnmix.build_model(model_path)
        package = torch.load(model_path, map_location=lambda storage, loc: storage)
        modules[source].load_state_dict(package['state_dict'])

    model = ParallelOpenUnmix(modules)
    
    return model

def apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude, iteration=1, channels_first=True, eps=EPS):
    """
    Multichannel Wiener filter.
    Implementation is based on norbert package.
    Args:
        mixture <torch.Tensor>: Complex tensor with shape of (1, n_channels, n_bins, n_frames) or (n_channels, n_bins, n_frames) or (batch_size, 1, n_channels, n_bins, n_frames) or (batch_size, n_channels, n_bins, n_frames)
        estimated_sources_amplitude <torch.Tensor>: (n_sources, n_channels, n_bins, n_frames) or (batch_size, n_sources, n_channels, n_bins, n_frames)
        iteration <int>: Iteration of EM algorithm updates
        channels_first <bool>: Only supports True
        eps <float>: small value for numerical stability
    """
    return multichannel_wiener_filter(mixture, estimated_sources_amplitude, iteration=iteration, channels_first=channels_first, eps=eps)