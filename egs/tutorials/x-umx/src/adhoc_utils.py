import os

import torch
import torchaudio
import torch.nn.functional as F

from algorithm.frequency_mask import compute_ideal_ratio_mask, multichannel_wiener_filter
from models.xumx import CrossNetOpenUnmix

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
NUM_CHANNELS_MUSDB18 = 2
BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

def separate_by_xumx(model_path, file_paths, out_dirs):
    use_cuda = torch.cuda.is_available()

    model = load_pretrained_model(model_path)
    config = load_experiment_config(model_path)
    
    sources = config['sources']
    patch_size = config['patch_size']
    n_fft, hop_length = config['n_fft'], config['hop_length']
    window = torch.hann_window(n_fft)
    
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

        if sample_rate == config['sample_rate']:
            pre_resampler, post_resampler = None, None
        else:
            pre_resampler, post_resampler = torchaudio.transforms.Resample(sample_rate, config['sample_rate']), torchaudio.transforms.Resample(config['sample_rate'], sample_rate)

        if pre_resampler is not None:
            x = pre_resampler(x)
        
        mixture = torch.stft(x, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
        padding = (patch_size - mixture.size(-1) % patch_size) % patch_size

        mixture = F.pad(mixture, (0, padding))
        mixture = mixture.reshape(*mixture.size()[:2], -1, patch_size)
        mixture = mixture.permute(2, 0, 1, 3).unsqueeze(dim=1)

        if use_cuda:
            mixture = mixture.cuda()

        n_sources = len(sources)

        with torch.no_grad():
            batch_size, _, n_mics, n_bins, n_frames = mixture.size()
            
            mixture_amplitude = torch.abs(mixture)
            
            estimated_sources_amplitude = []

            # Serial operation
            for _mixture_amplitude in mixture_amplitude:
                # _mixture_amplitude: (1, n_mics, n_bins, n_frames)
                if n_mics == 1:
                    _mixture_amplitude = torch.tile(_mixture_amplitude, (1, NUM_CHANNELS_MUSDB18, 1, 1))
                elif n_mics == 2:
                    _mixture_amplitude_flipped = torch.flip(_mixture_amplitude, dims=(1,))
                    _mixture_amplitude = torch.cat([_mixture_amplitude, _mixture_amplitude_flipped], dim=0)
                else:
                    raise NotImplementedError("Not support {} channels input.".format(n_mics))
                
                _mixture_amplitude = _mixture_amplitude.unsqueeze(dim=1) # (n_flips, 1, n_mics, n_bins, n_frames)
                _estimated_sources_amplitude = model(_mixture_amplitude) # (n_flips, n_sources, n_mics, n_bins, n_frames)

                if n_mics == 1:
                    _estimated_sources_amplitude = _estimated_sources_amplitude.mean(dim=2, keepdim=True) # (1, n_sources, n_mics, n_bins, n_frames)
                elif n_mics == 2:
                    _estimated_sources_amplitude, _estimated_sources_amplitude_flipped = torch.unbind(_estimated_sources_amplitude, dim=0) # n_flips of (n_sources, n_mics, n_bins, n_frames)
                    _estimated_sources_amplitude_flipped = torch.flip(_estimated_sources_amplitude_flipped, dims=(1,)) # (n_sources, n_mics, n_bins, n_frames)
                    _estimated_sources_amplitude = torch.stack([_estimated_sources_amplitude, _estimated_sources_amplitude_flipped], dim=0) # (n_flips, n_sources, n_mics, n_bins, n_frames)
                    _estimated_sources_amplitude = _estimated_sources_amplitude.mean(dim=0, keepdim=True) # (1, n_sources, n_mics, n_bins, n_frames)
                else:
                    raise NotImplementedError("Not support {} channels input.".format(n_mics))
                
                estimated_sources_amplitude.append(_estimated_sources_amplitude)

            estimated_sources_amplitude = torch.cat(estimated_sources_amplitude, dim=0) # (batch_size, n_sources, n_mics, n_bins, n_frames)
            estimated_sources_amplitude = estimated_sources_amplitude.permute(1, 2, 3, 0, 4) # (n_sources, n_mics, n_bins, batch_size, n_frames)
            estimated_sources_amplitude = estimated_sources_amplitude.reshape(n_sources, n_mics, n_bins, batch_size * n_frames) # (n_sources, n_mics, n_bins, batch_size * n_frames)
            estimated_sources_amplitude = estimated_sources_amplitude.cpu()

            mixture = mixture.permute(1, 2, 3, 0, 4).reshape(1, n_mics, n_bins, batch_size * n_frames) # (1, n_mics, n_bins, batch_size * n_frames)
            mixture = mixture.cpu()

            if n_mics == 1:
                estimated_sources_amplitude = estimated_sources_amplitude.squeeze(dim=1) # (n_sources, n_bins, batch_size * n_frames)
                mask = compute_ideal_ratio_mask(estimated_sources_amplitude)
                mask = mask.unsqueeze(dim=1) # (n_sources, n_mics, n_bins, batch_size * n_frames)
                estimated_sources = mask * mixture # (n_sources, n_mics, n_bins, batch_size * n_frames)
            else:
                estimated_sources = apply_multichannel_wiener_filter_torch(mixture, estimated_sources_amplitude=estimated_sources_amplitude)
            
            estimated_sources_channels = estimated_sources.size()[:-2]

            estimated_sources = estimated_sources.view(-1, *estimated_sources.size()[-2:])
            y = torch.istft(estimated_sources, n_fft, hop_length=hop_length, window=window, return_complex=False)
            
            if post_resampler is not None:
                y = post_resampler(y)
            
            y = y.view(*estimated_sources_channels, -1) # -> (n_sources, n_mics, T_pad)
            T_pad = y.size(-1)
            y = F.pad(y, (0, T_original - T_pad)) # -> (n_sources, n_mics, T_original)

            os.makedirs(out_dir, exist_ok=True)
            _estimated_paths = {}

            for idx in range(n_sources):
                source = sources[idx]
                path = os.path.join(out_dir, "{}.wav".format(source))
                torchaudio.save(path, y[idx], sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                _estimated_paths[source] = path
            
            estimated_paths.append(_estimated_paths)
    
    return estimated_paths

def load_pretrained_model(model_path):
    model = CrossNetOpenUnmix.build_model(model_path, load_state_dict=True)
    
    return model

def load_experiment_config(config_path):
    config = torch.load(config_path, map_location=lambda storage, loc: storage)

    config = {
        'sample_rate': config.get('sample_rate') or SAMPLE_RATE_MUSDB18,
        'sources': config['sources'],
        'patch_size': config.get('patch_size') or 256,
        'n_fft': config.get('n_fft') or 4096,
        'hop_length': config.get('hop_length') or 1024
    }

    return config

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