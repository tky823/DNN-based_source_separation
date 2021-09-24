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

        if use_cuda:
            x = x.cuda()

        with torch.no_grad():
            n_mics = x.size(0)
            mixture = x.view(1, 1, n_mics, -1)

            if n_mics == 1:
                mixture = torch.tile(mixture, (1, 1, NUM_CHANNELS_MUSDB18, 1))
            estimated_sources = model(mixture)
            if n_mics == 1:
                mixture = mixture.mean(dim=2, keepdim=True)

            mixture = mixture.cpu()
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

            n_sources = len(__sources__)

            for idx in range(n_sources):
                source = __sources__[idx]
                path = os.path.join(out_dir, "{}.wav".format(source))
                torchaudio.save(path, y[idx], sample_rate=sample_rate, bits_per_sample=BITS_PER_SAMPLE_MUSDB18)
                _estimated_paths[source] = path
            
            estimated_paths.append(_estimated_paths)
            
    return estimated_paths

def load_pretrained_conv_tasnet(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    model = ConvTasNet.build_model(model_path)
    model.load_state_dict(package['state_dict'])
    
    return model