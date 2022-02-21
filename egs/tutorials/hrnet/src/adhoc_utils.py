import os

import torch
import torchaudio
import torch.nn.functional as F

from models.hrnet import HRNet

SAMPLE_RATE_MUSDB18_HRNET = 16000
NUM_CHANNELS_MUSDB18 = 2
BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

def separate_by_hrnet(model_paths, file_paths, out_dirs):
    use_cuda = torch.cuda.is_available()

    model = load_pretrained_model(model_paths)
    config = load_experiment_config(model_paths)

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
        mixture = mixture.permute(2, 0, 1, 3)

        if use_cuda:
            mixture = mixture.cuda()

        with torch.no_grad():
            batch_size, n_mics, n_bins, n_frames = mixture.size()
            
            mixture_amplitude = torch.abs(mixture)
            
            estimated_source_amplitude = []

            # Serial operation
            for _mixture_amplitude in mixture_amplitude:
                # _mixture_amplitude: (n_mics, n_bins, n_frames)
                if n_mics == 1:
                    _mixture_amplitude = torch.tile(_mixture_amplitude, (1, NUM_CHANNELS_MUSDB18, 1, 1))
                elif n_mics == 2:
                    _mixture_amplitude_flipped = torch.flip(_mixture_amplitude, dims=(1,))
                    _mixture_amplitude = torch.stack([_mixture_amplitude, _mixture_amplitude_flipped], dim=0)
                else:
                    raise NotImplementedError("Not support {} channels input.".format(n_mics))
                
                _estimated_source_amplitude = model(_mixture_amplitude)

                if n_mics == 1:
                    _estimated_source_amplitude = _estimated_source_amplitude.mean(dim=1)
                elif n_mics == 2:
                    sections = [1, 1]
                    _estimated_source_amplitude, _estimated_source_amplitude_flipped = torch.split(_estimated_source_amplitude, sections, dim=0)
                    _estimated_source_amplitude_flipped = torch.flip(_estimated_source_amplitude_flipped, dims=(1,))
                    _estimated_source_amplitude = torch.cat([_estimated_source_amplitude, _estimated_source_amplitude_flipped], dim=0)
                    _estimated_source_amplitude = _estimated_source_amplitude.mean(dim=0)
                else:
                    raise NotImplementedError("Not support {} channels input.".format(n_mics))
                
                estimated_source_amplitude.append(_estimated_source_amplitude)

            estimated_source_amplitude = torch.stack(estimated_source_amplitude, dim=0) # (batch_size, n_mics, n_bins, n_frames)
            estimated_source_amplitude = estimated_source_amplitude.permute(1, 2, 0, 3)
            estimated_source_amplitude = estimated_source_amplitude.reshape(n_mics, n_bins, batch_size * n_frames) # (n_mics, n_bins, batch_size * n_frames)

            mixture = mixture.permute(1, 2, 0, 3).reshape(n_mics, n_bins, batch_size * n_frames) # (n_mics, n_bins, batch_size * n_frames)

            mixture = mixture.cpu()
            estimated_source_amplitude = estimated_source_amplitude.cpu()

            mixture_phase = torch.angle(mixture)
            estimated_source = estimated_source_amplitude * torch.exp(1j * mixture_phase)
            estimated_source_channels = estimated_source.size()[:-2]

            estimated_source = estimated_source.view(-1, *estimated_source.size()[-2:])
            y = torch.istft(estimated_source, n_fft, hop_length=hop_length, window=window, return_complex=False)
            
            if post_resampler is not None:
                y = post_resampler(y)
            
            y = y.view(*estimated_source_channels, -1) # -> (n_mics, T_pad)
            T_pad = y.size(-1)
            y = F.pad(y, (0, T_original - T_pad)) # -> (n_mics, T_original)

            os.makedirs(out_dir, exist_ok=True)

            _estimated_path = os.path.join(out_dir, "separated.wav")
            torchaudio.save(_estimated_path, y, sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
            
            estimated_paths.append(_estimated_path)
            
    return estimated_paths

def load_pretrained_model(model_path):
    model = HRNet.build_model(model_path, load_state_dict=True)
    
    return model

def load_experiment_config(config_path):
    config = torch.load(config_path, map_location=lambda storage, loc: storage)

    sample_rate = config.get('sample_rate')
    patch_size = config.get('patch_size')
    n_fft = config.get('n_fft')
    hop_length = config.get('hop_length')
    
    config = {
        'sample_rate': sample_rate or SAMPLE_RATE_MUSDB18_HRNET,
        'patch_size': patch_size or 64,
        'n_fft': n_fft or 1024,
        'hop_length': hop_length or 512
    }

    return config
