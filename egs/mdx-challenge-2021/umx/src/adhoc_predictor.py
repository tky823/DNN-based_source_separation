import os
import time

import torch
import torchaudio
import torch.nn.functional as F

from evaluator.music_demixing import MusicDemixingPredictor
from utils.audio import build_window
from algorithm.frequency_mask import multichannel_wiener_filter
from models.umx import OpenUnmix, ParallelOpenUnmix

__sources__ = ['bass', 'drums', 'other', 'vocals']
BITS_PER_SAMPLE = 16
EPS = 1e-12

def separate(waveform, umx, n_fft=4096, hop_length=1024, window_fn='hann', patch_size=256, sources=__sources__, iteration_wfm=1, device="cpu"):
    """
    Args:
        waveform <torch.Tensor>: Mixture waveform with shape of (2, T).
        umx <models.ParallelOpenUnmix>: Pretrained model.
        n_fft <int>: Default: 4096
        hop_length <int>: Default: 1024
        window_fn <str>: Window function. Default: 'hann'
        patch_size <int>: Default: 256
        sources <list<str>>: Target sources.
        iteration_wfm <int>: Iterations of Wiener Filter Mask.
        device <str>: Only supports "cpu".
    Returns:
        estimates <dict<torch.Tensor>>: All estimates obtained by the separation model.
    """
    window = build_window(n_fft, window_fn=window_fn)

    n_mics, T = waveform.size()
    mixture = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    padding = (patch_size - mixture.size(-1) % patch_size) % patch_size

    mixture = F.pad(mixture, (0, padding))
    mixture = mixture.reshape(*mixture.size()[:2], -1, patch_size)
    mixture = mixture.permute(2, 0, 1, 3).unsqueeze(dim=1)
    mixture = mixture.to(device)

    n_sources = len(sources)

    with torch.no_grad():
        batch_size, _, _, n_bins, n_frames = mixture.size()

        mixture_amplitude = torch.abs(mixture)

        estimated_sources_amplitude = {
            target: [] for target in sources
        }

        # Serial operation
        for _mixture_amplitude in mixture_amplitude:
            # _mixture_amplitude: (1, n_mics, n_bins, n_frames)
            for target in sources:
                _estimated_source_amplitude = umx(_mixture_amplitude, target=target)
                estimated_sources_amplitude[target].append(_estimated_source_amplitude)

        estimated_sources_amplitude = [
            torch.cat(estimated_sources_amplitude[target], dim=0).unsqueeze(dim=0) for target in sources
        ]
        estimated_sources_amplitude = torch.cat(estimated_sources_amplitude, dim=0) # (n_sources, batch_size, n_mics, n_bins, n_frames)
        estimated_sources_amplitude = estimated_sources_amplitude.permute(0, 2, 3, 1, 4).reshape(n_sources, n_mics, n_bins, batch_size * n_frames) # (n_sources, n_mics, n_bins, batch_size * n_frames)
        mixture = mixture.permute(1, 2, 3, 0, 4).reshape(1, n_mics, n_bins, batch_size * n_frames) # (1, n_mics, n_bins, batch_size * n_frames)

        mixture, estimated_sources_amplitude = F.pad(mixture, (0, - padding)), F.pad(estimated_sources_amplitude, (0, - padding)) # (1, n_mics, n_bins, batch_size * n_frames - padding), (n_sources, n_mics, n_bins, batch_size * n_frames - padding)
        mixture, estimated_sources_amplitude = mixture.cpu(), estimated_sources_amplitude.cpu()
        estimated_sources = multichannel_wiener_filter(mixture, estimated_sources_amplitude=estimated_sources_amplitude, iteration=iteration_wfm) # (n_sources, n_mics, n_bins, batch_size * n_frames - padding)

        estimated_sources = estimated_sources.view(-1, n_bins, batch_size * n_frames - padding) # (n_sources * n_mics, n_bins, batch_size * n_frames - padding)
        estimated_waveforms = torch.istft(estimated_sources, n_fft, hop_length=hop_length, window=window, return_complex=False) # (n_sources * n_mics, n_bins, T_pad)
        estimated_waveforms = estimated_waveforms.view(n_sources, n_mics, -1) # (n_sources, n_mics, T_pad)
        T_pad = estimated_waveforms.size(-1)
        estimated_waveforms = F.pad(estimated_waveforms, (0, T - T_pad)) # (n_sources, n_mics, T)

    estimates = {}

    for target, estimated_waveform in zip(sources, estimated_waveforms):
        estimates[target] = estimated_waveform # (n_mics, T)

    return estimates

class UMXPredictor(MusicDemixingPredictor):
    def __init__(self, args):
        super().__init__()

        self.patch_size = args.patch_size
        self.n_fft, self.hop_length = args.n_fft, args.hop_length
        self.window_fn = args.window_fn

        self.sources = args.sources
        self.model_dir = args.model_dir

    def prediction_setup(self):
        modules = {}

        for source in self.sources:
            model_path = os.path.join(self.model_dir, "{}.pth".format(source))
            if not os.path.exists(model_path):
                raise FileNotFoundError("Cannot find {}.".format(model_path))
            modules[source] = OpenUnmix.build_model(model_path, load_state_dict=True)

        self.separator = ParallelOpenUnmix(modules)
        self.separator.eval()

    def prediction(self, mixture_file_path, bass_file_path, drums_file_path, other_file_path, vocals_file_path):
        start = time.time()

        # Step 1: Load mixture
        waveform, rate = torchaudio.load(mixture_file_path)

        # Step 2: Perform separation (includes pad and crop)
        estimates = separate(waveform, self.separator, n_fft=self.n_fft, hop_length=self.hop_length, window_fn=self.window_fn, patch_size=self.patch_size, sources=self.sources)

        # Step 3: Store results
        target_file_map = {
            "vocals": vocals_file_path,
            "drums": drums_file_path,
            "bass": bass_file_path,
            "other": other_file_path,
        }
        for target, path in target_file_map.items():
            torchaudio.save(path, estimates[target], rate, bits_per_sample=BITS_PER_SAMPLE)

        end = time.time()

        print("{:.3f} [sec]".format(end - start))
