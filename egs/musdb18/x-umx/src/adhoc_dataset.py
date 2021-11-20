import os

import torch
import torchaudio
import torch.nn.functional as F

from transforms.stft import stft
from dataset import assert_sample_rate
from dataset import SpectrogramDataset

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, patch_size=256, max_samples=None, sources=__sources__, target=None):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)
        
        assert_sample_rate(sample_rate)

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
            track_sample_rate = audio_info.sample_rate
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
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            name <str>: Artist and title of track
        """
        track_data = self.json_data[idx]
        patch_size = self.patch_size

        trackID = track_data['trackID']
        track = self.tracks[trackID]
        name = track['name']
        paths = track['path']
        samples = track_data['samples']

        if set(self.sources) == set(__sources__):
            mixture, _ = torchaudio.load(paths['mixture'], num_frames=samples) # (n_mics, T)
        else:
            sources = []
            for _source in self.sources:
                source, _ = torchaudio.load(paths[_source], num_frames=samples) # (n_mics, T)
                sources.append(source.unsqueeze(dim=0))
            sources = torch.cat(sources, dim=0) # (len(self.sources), n_mics, T)
            mixture = sources.sum(dim=0) # (n_mics, T)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                source, _ = torchaudio.load(paths[_target], num_frames=samples) # (n_mics, T)
                target.append(source.unsqueeze(dim=0))
            target = torch.cat(target, dim=0) # (len(target), n_mics, T)
            mixture = mixture.unsqueeze(dim=0) # (1, n_mics, T)
        else:
            # mixture: (n_mics, T)
            target, _ = torchaudio.load(paths[self.target], num_frames=samples) # (n_mics, T)

        mixture_channels, target_channels = mixture.size()[:-1], target.size()[:-1]
        mixture = mixture.reshape(-1, mixture.size(-1))
        target = target.reshape(-1, target.size(-1))

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        target = stft(target, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        
        n_frames = mixture.size(-1)
        padding = (patch_size - n_frames % patch_size) % patch_size

        mixture = F.pad(mixture, (0, padding))
        target = F.pad(target, (0, padding))

        mixture = mixture.reshape(*mixture.size()[:-1], -1, patch_size)
        target = target.reshape(*target.size()[:-1], -1, patch_size)

        mixture = mixture.permute(2, 0, 1, 3).contiguous() # (batch_size, n_mics, n_bins, n_frames)
        target = target.permute(2, 0, 1, 3).contiguous() # (batch_size, len(target) * n_mics, n_bins, n_frames) or (batch_size, n_mics, n_bins, n_frames)

        mixture = mixture.reshape(-1, *mixture_channels, *mixture.size()[-2:]) # (batch_size, 1, n_mics, n_bins, n_frames) or # (batch_size, n_mics, n_bins, n_frames)
        target = target.reshape(-1, *target_channels, *target.size()[-2:]) # (batch_size, len(target), n_mics, n_bins, n_frames) or (batch_size, n_mics, n_bins, n_frames)
        
        return mixture, target, name

class SpectrogramTestDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, patch_size=256, sources=__sources__, target=None):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)

        assert_sample_rate(sample_rate)

        test_txt_path = os.path.join(musdb18_root, 'test.txt')

        names = []
        with open(test_txt_path, 'r') as f:
            for line in f:
                name = line.strip()
                names.append(name)
        
        self.patch_size = patch_size

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'test', name, "mixture.wav")
            audio_info = torchaudio.info(mixture_path)
            track_sample_rate = audio_info.sample_rate
            track_samples = audio_info.num_frames

            track = {
                'name': name,
                'samples': track_samples,
                'path': {
                    'mixture': mixture_path
                }
            }

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'test', name, "{}.wav".format(source))
            
            track_data = {
                'trackID': trackID,
                'start': 0,
                'samples': track_samples
            }
            
            self.tracks.append(track)
            self.json_data.append(track_data) # len(self.json_data) determines # of samples in dataset
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            samples <int>: Number of samples in time domain.
            name <str>: Artist and title of track
        """
        track_data = self.json_data[idx]
        patch_size = self.patch_size

        trackID = track_data['trackID']
        track = self.tracks[trackID]
        name = track['name']
        paths = track['path']
        samples = track['samples']

        if set(self.sources) == set(__sources__):
            mixture, _ = torchaudio.load(paths['mixture']) # (n_mics, T)
        else:
            sources = []
            for _source in self.sources:
                source, _ = torchaudio.load(paths[_source]) # (n_mics, T)
                sources.append(source.unsqueeze(dim=0))
            sources = torch.cat(sources, dim=0) # (len(self.sources), n_mics, T)
            mixture = sources.sum(dim=0) # (n_mics, T)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                source, _ = torchaudio.load(paths[_target]) # (n_mics, T)
                target.append(source.unsqueeze(dim=0))
            target = torch.cat(target, dim=0) # (len(target), n_mics, T)
            mixture = mixture.unsqueeze(dim=0) # (1, n_mics, T)
        else:
            # mixture: (n_mics, T)
            target, _ = torchaudio.load(paths[self.target]) # (n_mics, T)

        mixture_channels, target_channels = mixture.size()[:-1], target.size()[:-1]
        mixture = mixture.reshape(-1, mixture.size(-1))
        target = target.reshape(-1, target.size(-1))

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        target = stft(target, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        
        n_frames = mixture.size(-1)
        padding = (patch_size - n_frames % patch_size) % patch_size

        mixture = F.pad(mixture, (0, padding))
        target = F.pad(target, (0, padding))

        mixture = mixture.reshape(*mixture.size()[:-1], -1, patch_size)
        target = target.reshape(*target.size()[:-1], -1, patch_size)

        mixture = mixture.permute(2, 0, 1, 3).contiguous() # (batch_size, n_mics, n_bins, n_frames)
        target = target.permute(2, 0, 1, 3).contiguous() # (batch_size, len(target) * n_mics, n_bins, n_frames) or (batch_size, n_mics, n_bins, n_frames)

        mixture = mixture.reshape(-1, *mixture_channels, *mixture.size()[-2:]) # (batch_size, 1, n_mics, n_bins, n_frames) or # (batch_size, n_mics, n_bins, n_frames)
        target = target.reshape(-1, *target_channels, *target.size()[-2:]) # (batch_size, len(target), n_mics, n_bins, n_frames) or (batch_size, n_mics, n_bins, n_frames)
        
        return mixture, target, samples, name

"""
Data loader
"""
class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

        self.collate_fn = eval_collate_fn

class TestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

        self.collate_fn = test_collate_fn

def eval_collate_fn(batch):
    mixture, sources, name = batch[0]
    
    return mixture, sources, name

def test_collate_fn(batch):
    mixture, sources, samples, name = batch[0]
    
    return mixture, sources, samples, name
