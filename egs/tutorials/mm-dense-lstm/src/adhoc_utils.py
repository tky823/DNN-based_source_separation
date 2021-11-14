import os

import torch
import torchaudio
import torch.nn.functional as F

from algorithm.frequency_mask import ideal_ratio_mask, multichannel_wiener_filter
from models.mm_dense_lstm import MMDenseLSTM, ParallelMMDenseLSTM

__sources__ = ['bass', 'drums', 'other', 'vocals']
SAMPLE_RATE_MUSDB18 = 44100
NUM_CHANNELS_MUSDB18 = 2
BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

def separate_by_mm_dense_lstm(model_paths, file_paths, out_dirs):
    use_cuda = torch.cuda.is_available()

    model = load_pretrained_model(model_paths)
    config = load_experiment_config(model_paths)

    patch_size = config['patch_size']
    fft_size, hop_size = config['fft_size'], config['hop_size']
    window = torch.hann_window(fft_size)
    
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
                elif n_mics == 2:
                    _mixture_amplitude_flipped = torch.flip(_mixture_amplitude, dims=(1,))
                    _mixture_amplitude = torch.cat([_mixture_amplitude, _mixture_amplitude_flipped], dim=0)
                else:
                    raise NotImplementedError("Not support {} channels input.".format(n_mics))
                
                for target in __sources__:
                    _estimated_sources_amplitude = model(_mixture_amplitude, target=target)

                    if n_mics == 1:
                        _estimated_sources_amplitude = _estimated_sources_amplitude.mean(dim=1, keepdim=True)
                    elif n_mics == 2:
                        sections = [1, 1]
                        _estimated_sources_amplitude, _estimated_sources_amplitude_flipped = torch.split(_estimated_sources_amplitude, sections, dim=0)
                        _estimated_sources_amplitude_flipped = torch.flip(_estimated_sources_amplitude_flipped, dims=(1,))
                        _estimated_sources_amplitude = torch.cat([_estimated_sources_amplitude, _estimated_sources_amplitude_flipped], dim=0)
                        _estimated_sources_amplitude = _estimated_sources_amplitude.mean(dim=0, keepdim=True)
                    else:
                        raise NotImplementedError("Not support {} channels input.".format(n_mics))
                    
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

def load_pretrained_model(model_paths):
    modules = {}

    for source in __sources__:
        model_path = model_paths[source]
        modules[source] = MMDenseLSTM.build_model(model_path, load_state_dict=True)

    model = ParallelMMDenseLSTM(modules)
    
    return model

def load_experiment_config(config_paths):
    sample_rate = None
    patch_size = None
    fft_size, hop_size = None, None

    for source in __sources__:
        config_path = config_paths[source]
        config = torch.load(config_path, map_location=lambda storage, loc: storage)

        if sample_rate is None:
            sample_rate = config.get('sample_rate')
        elif config.get('sample_rate') is not None:
            if sample_rate is not None:
                assert sample_rate == config['sample_rate'], "Invalid sampling rate."
            sample_rate = config['sample_rate']
        
        if patch_size is None:
            patch_size = config.get('patch_size')
        elif config.get('patch_size') is not None:
            if patch_size is not None:
                assert patch_size == config['patch_size'], "Invalid patch_size."
            patch_size = config['patch_size']
        
        if fft_size is None:
            fft_size = config.get('fft_size')
        elif config.get('fft_size') is not None:
            if fft_size is not None:
                assert fft_size == config['fft_size'], "Invalid fft_size."
            fft_size = config['fft_size']

        if hop_size is None:
            hop_size = config.get('hop_size')
        elif config.get('hop_size') is not None:
            if hop_size is not None:
                assert hop_size == config['hop_size'], "Invalid hop_size."
            hop_size = config['hop_size']
    
    config = {
        'sample_rate': sample_rate or SAMPLE_RATE_MUSDB18,
        'patch_size': patch_size or 256,
        'fft_size': fft_size or 4096,
        'hop_size': hop_size or 1024
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