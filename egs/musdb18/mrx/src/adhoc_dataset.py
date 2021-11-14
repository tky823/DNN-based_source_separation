import os

import torch
import torchaudio
import torch.nn.functional as F

from dataset import assert_sample_rate
from dataset import WaveDataset

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, patch_duration=6, max_duration=None, sources=__sources__, target=None):
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)
        
        assert_sample_rate(sample_rate)

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        
        with open(valid_txt_path, 'r') as f:
            names = [line.strip() for line in f]

        self.sample_rate = sample_rate
        self.patch_duration = patch_duration
        max_samples = int(sample_rate * max_duration)

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
            audio_info = torchaudio.info(mixture_path)
            track_sample_rate = audio_info.sample_rate
            track_samples = audio_info.num_frames
            samples = min(max_samples, track_samples)

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
            mixture <torch.Tensor>: Complex tensor with shape (batch_size, 1, n_mics, patch_samples)  if `target` is list, otherwise (batch_size, n_mics, patch_samples) 
            target <torch.Tensor>: Complex tensor with shape (batch_size, len(target), n_mics, patch_samples) if `target` is list, otherwise (batch_size, n_mics, patch_samples)
            name <str>: Artist and title of track
        """
        patch_samples = int(self.sample_rate * self.patch_duration)
        track_data = self.json_data[idx]

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
        
        padding = (patch_samples - mixture.size(-1) % patch_samples) % patch_samples
        mixture = F.pad(mixture, (0, padding))
        target = F.pad(target, (0, padding))

        mixture_channels = mixture.size()[:-1]
        target_channels = target.size()[:-1]

        mixture = mixture.reshape(*mixture.size()[:-1], -1, patch_samples)
        target = target.reshape(*target.size()[:-1], -1, patch_samples)

        mixture = mixture.permute(2, 0, 1, 3).contiguous() # (batch_size, n_mics, patch_samples)
        target = target.permute(2, 0, 1, 3).contiguous() # (batch_size, len(target) * n_mics, patch_samples) or (batch_size, n_mics, patch_samples)

        mixture = mixture.reshape(-1, *mixture_channels, *mixture.size()[-1:]) # (batch_size, 1, n_mics, patch_samples) or # (batch_size, n_mics, patch_samples)
        target = target.reshape(-1, *target_channels, *target.size()[-1:]) # (batch_size, len(target), n_mics, patch_samples) or (batch_size, n_mics, patch_samples)

        return mixture, target, name

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
