import os

import torch
import torchaudio
import torch.nn.functional as F

from models.conv_tasnet import ConvTasNet

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
NUM_CHANNELS_MUSDB18 = 2
BITS_PER_SAMPLE_MUSDB18 = 16
EPS = 1e-12

def separate_by_conv_tasnet(model_path, file_paths, out_dirs):
    use_cuda = torch.cuda.is_available()

    model = load_pretrained_conv_tasnet(model_path)
    config = load_experiment_config(model_path)

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

        if use_cuda:
            x = x.cuda()

        with torch.no_grad():
            n_mics = x.size(0)
            mixture = x.view(1, 1, n_mics, -1)

            if n_mics == 1:
                mixture = torch.tile(mixture, (1, 1, NUM_CHANNELS_MUSDB18, 1))
            elif n_mics == 2:
                mixture_flipped = torch.flip(mixture, dims=(2,))
                mixture = torch.cat([mixture, mixture_flipped], dim=0)
            else:
                raise NotImplementedError("Not support {} channels input.".format(n_mics))

            mean, std = mixture.mean(dim=-1, keepdim=True), mixture.std(dim=-1, keepdim=True)
            standardized_mixture = (mixture - mean) / (std + EPS)
            standardized_estimated_sources = model(standardized_mixture)
            estimated_sources = std * standardized_estimated_sources + mean

            if n_mics == 1:
                estimated_sources = estimated_sources.mean(dim=2, keepdim=True)
            elif n_mics == 2:
                sections = [1, 1]
                estimated_sources, estimated_sources_flipped = torch.split(estimated_sources, sections, dim=0)
                estimated_sources_flipped = torch.flip(estimated_sources_flipped, dims=(2,))
                estimated_sources = torch.cat([estimated_sources, estimated_sources_flipped], dim=0)
                estimated_sources = estimated_sources.mean(dim=0, keepdim=True)
            else:
                raise NotImplementedError("Not support {} channels input.".format(n_mics))

            max_value = torch.max(torch.abs(estimated_sources))
            max_value = max_value.item()

            if max_value >= 1:
                estimated_sources = 0.9 * (estimated_sources / max_value)

            estimated_sources = estimated_sources.cpu()

            estimated_sources_channels = estimated_sources.size()[:-1]
            y = estimated_sources.view(-1, estimated_sources.size(-1))

            if post_resampler is not None:
                y = post_resampler(y)

            y = y.view(*estimated_sources_channels, -1) # -> (n_mics, n_sources, T_pad)
            y = y.squeeze(dim=0) # -> (n_sources, n_mics, T_pad)
            T_pad = y.size(-1)
            y = F.pad(y, (0, T_original - T_pad)) # -> (n_sources, n_mics, T_original)

            os.makedirs(out_dir, exist_ok=True)
            _estimated_paths = {}

            for target, estimated_source in zip(config['sources'], y):
                path = os.path.join(out_dir, "{}.wav".format(target))
                torchaudio.save(path, estimated_source, sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                _estimated_paths[target] = path

            estimated_paths.append(_estimated_paths)

    return estimated_paths

def load_pretrained_conv_tasnet(model_path):
    model = ConvTasNet.build_model(model_path, load_state_dict=True)

    return model

def load_experiment_config(config_path):
    config = torch.load(config_path, map_location=lambda storage, loc: storage)
    config = {
        'sample_rate': config.get('sr') or config.get('sample_rate') or SAMPLE_RATE_MUSDB18,
        'sources': config.get('sources') or __sources__
    }

    return config