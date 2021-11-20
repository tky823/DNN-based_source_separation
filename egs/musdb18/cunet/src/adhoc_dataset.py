import os
import random

import torch
import torchaudio

from utils.utils_audio import build_window
from transforms.stft import stft
from dataset import MUSDB18Dataset

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12
THRESHOLD_POWER = 1e-5

class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sample_rate=44100, sources=__sources__, target=None):
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)

        self.json_data = None

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (n_mics, T)
            target <torch.Tensor>: (n_mics, T)
            latent <torch.Tensor>: (len(target),)
            name <str>: Artist and title of track
            sources <torch.Tensor>: (len(target),n_mics, T)
            scale <float>: ()
        """
        data = self.json_data[idx]

        trackID = data['trackID']
        track = self.tracks[trackID]
        name = track['name']
        paths = track['path']
        start = data['start']
        samples = data['samples']

        if set(self.sources) == set(__sources__):
            mixture, _ = torchaudio.load(paths['mixture'], frame_offset=start, num_frames=samples)
        else:
            sources = []
            for _source in self.sources:
                source, _ = torchaudio.load(paths[_source], frame_offset=start, num_frames=samples)
                sources.append(source.unsqueeze(dim=0))
            sources = torch.cat(sources, dim=0)
            mixture = sources.sum(dim=0)
        
        if type(self.target) is list:
            latent = torch.zeros(len(self.target))

            _target = random.choice(self.sources)
            source_idx = self.sources.index(_target)
            scale = random.uniform(0, 1)
            latent[source_idx] = scale

            target, _ = torchaudio.load(paths[_target], frame_offset=start, num_frames=samples)
            target = scale * target
        else:
            raise ValueError("self.target must be list.")

        return mixture, target, latent, name, scale

    def __len__(self):
        return len(self.json_data)

class SpectrogramDataset(WaveDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=44100, sources=__sources__, target=None):
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)
        
        if hop_length is None:
            hop_length = n_fft // 2
        
        self.n_fft, self.hop_length = n_fft, hop_length
        self.n_bins = n_fft // 2 + 1

        if window_fn:
            self.window = build_window(n_fft, window_fn=window_fn)
        else:
            self.window = None
        
        self.normalize = normalize

    def _is_active(self, input, threshold=THRESHOLD_POWER):
        n_dims = input.dim()

        if n_dims > 2:
            input = input.reshape(-1, input.size(-1))

        input = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources)*2, n_bins, n_frames)
        power = torch.sum(torch.abs(input)**2, dim=-1) # (len(sources)*2, n_bins, n_frames)
        power = torch.mean(power)

        if power.item() >= threshold:
            return True
        else:
            return False
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
            latent
            T (), <int>: Number of samples in time-domain
            title <str>: Title of track
        """
        mixture, target, latent, title, scale = super().__getitem__(idx)
        
        T = mixture.size(-1)

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, 2, n_bins, n_frames) or (2, n_bins, n_frames)
        target = stft(target, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), 2, n_bins, n_frames) or (2, n_bins, n_frames)

        return mixture, target, latent, T, title, scale

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=44100, patch_samples=4*SAMPLE_RATE_MUSDB18, overlap=None, sources=__sources__, target=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)
        
        train_txt_path = os.path.join(musdb18_root, 'train.txt')

        with open(train_txt_path, 'r') as f:
            names = [line.strip() for line in f]
        
        if overlap is None:
            overlap = patch_samples // 2

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
            audio_info = torchaudio.info(mixture_path)
            sample_rate = audio_info.sample_rate
            track_samples = audio_info.num_frames

            track = {
                'name': name,
                'samples': track_samples,
                'path': {
                    'mixture': mixture_path
                }
            }

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))
            
            self.tracks.append(track)

            for start in range(0, track_samples, patch_samples - overlap):
                if start + patch_samples >= track_samples:
                    break
                data = {
                    'trackID': trackID,
                    'start': start,
                    'samples': patch_samples,
                }
                self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
        """
        mixture, target, latent, _, _, _ = super().__getitem__(idx)

        return mixture, target, latent

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=44100, patch_size=256, max_samples=10*SAMPLE_RATE_MUSDB18, sources=__sources__, target=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)
        
        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        
        with open(valid_txt_path, 'r') as f:
            names = [line.strip() for line in f]

        self.patch_size = patch_size
        self.max_samples = max_samples

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
            audio_info = torchaudio.info(mixture_path)
            sample_rate = audio_info.sample_rate
            track_samples = audio_info.num_frames
            samples = min(self.max_samples, track_samples)

            track = {
                'name': name,
                'samples': track_samples,
                'path': {
                    'mixture': mixture_path
                }
            }

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))
            
            track_data = {
                'trackID': trackID,
                'start': 0,
                'samples': samples
            }
            
            self.tracks.append(track)
            self.json_data.append(track_data) # len(self.json_data) determines # of samples in dataset
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
        """
        data = self.json_data[idx]

        trackID = data['trackID']
        track = self.tracks[trackID]
        paths = track['path']
        start, samples = data['start'], data['samples']

        sources = []
        target = []
        latent = torch.zeros((len(self.sources), len(self.sources)))
        scales = []
        source_names = self.sources.copy()

        for source_idx, source_name in enumerate(self.sources):
            path = paths[source_name]
            source, _ = torchaudio.load(path, frame_offset=start, num_frames=samples)
            sources.append(source)
            scale = random.uniform(0.5, 1) # 1 doesn't work.
            latent[source_idx, source_idx] = scale
            target.append(scale * source)
            scales.append(scale)
        
        sources = torch.concatenate(sources, dim=0)
        target = torch.concatenate(target, dim=0)
        mixture = sources.sum(dim=0, keepdim=True)

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, 2, n_bins, n_frames) or (2, n_bins, n_frames)
        target = stft(target, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), 2, n_bins, n_frames) or (2, n_bins, n_frames)

        return mixture, target, latent, source_names, scales

"""
Data loader
"""
class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

        self.collate_fn = eval_collate_fn

def eval_collate_fn(batch):
    mixture, target, latent, source_names, scale = batch[0]
    
    return mixture, target, latent, source_names, scale

def _test_train_dataset():
    torch.manual_seed(111)
    
    musdb18_root = "../../../../../db/musdb18"

    dataset = SpectrogramTrainDataset(musdb18_root, n_fft=2048, hop_length=512, sample_rate=8000, duration=4, target='vocals')
    
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

    dataset.save_as_json('data/tmp.json')

    dataset = SpectrogramTrainDataset.from_json(musdb18_root, 'data/tmp.json', n_fft=2048, hop_length=512, sample_rate=44100, target='vocals')
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

if __name__ == '__main__':
    _test_train_dataset()