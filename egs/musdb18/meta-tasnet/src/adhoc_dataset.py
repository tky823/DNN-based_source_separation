import random

import numpy as np
import musdb
import torch
import torch.nn.functional as F

from dataset import MUSDB18Dataset

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
SAMPLE_RATES_METATASNET = [8000, 16000, 32000]
EPS = 1e-12
THRESHOLD_POWER = 1e-5
MINSCALE = 0.75
MAXSCALE = 1.25

class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sources=__sources__, target=None, is_wav=False):
        """
        Args:
            musdb18_root <int>: Path to MUSDB or MUSDB-HQ
            sources <list<str>>: Sources included in mixture
            target <str> or <list<str>>: 
            is_wav <bool>
        """
        super().__init__(musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, sources=sources, target=target, is_wav=is_wav)

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
            name <str>: Artist and title of song
        """
        raise NotImplementedError("Implement __getitem__ in sub-class.")

    def __len__(self):
        raise NotImplementedError("Implement __len__ in sub-class.")

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, duration=8, samples_per_epoch=None, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, sources=sources, target=target, is_wav=is_wav)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train', is_wav=is_wav)
        
        self.duration = duration

        self.std = {}

        for songID, track in enumerate(self.mus.tracks):
            if set(self.sources) == set(__sources__):
                mixture = track.audio.transpose(1, 0)
            else:
                sources = []
                for _source in self.sources:
                    sources.append(track.targets[_source].audio.transpose(1, 0)[np.newaxis])
                sources = np.concatenate(sources, axis=0)
                mixture = sources.sum(axis=0)
            
            self.std[songID] = np.std(mixture.mean(axis=0))
        
        self.samples_per_epoch = samples_per_epoch

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, T) if `target` is list, otherwise (T,)
            target <torch.Tensor>: (len(target), T) if `target` is list, otherwise (T,)
        """
        n_songs = len(self.mus.tracks)
        song_indices = random.choices(range(n_songs), k=len(self.sources))

        sources = []
        songIDs = []
        starts = []
        channels = []
        scales = []

        for _source, songID in zip(self.sources, song_indices):
            track = self.mus.tracks[songID]
            std = self.std[songID]

            start = random.uniform(0, track.duration - self.duration)
            channel = random.randint(0, 1)
            scale = random.uniform(MINSCALE, MAXSCALE)

            track.chunk_start = start
            track.chunk_duration = self.duration

            source = track.targets[_source].audio.transpose(1, 0) / std
            source = source[channel]

            sources.append(scale * source[np.newaxis])
            songIDs.append(songID)
            starts.append(start)
            channels.append(channel)
            scales.append(scale)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                source_idx = self.sources.index(_target)
                _target = sources[source_idx]
                target.append(_target)
            target = np.concatenate(target, axis=0)

            sources = np.concatenate(sources, axis=0)
            mixture = sources.sum(axis=0, keepdims=True)
        else:
            source_idx = self.sources.index(self.target)
            target = sources[source_idx]
            target = target.squeeze(axis=0)

            sources = np.concatenate(sources, axis=0)
            mixture = sources.sum(axis=0)

        mixture, target = torch.Tensor(mixture).float(), torch.Tensor(target).float()

        return mixture, target

    def __len__(self):
        return self.samples_per_epoch

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, max_duration=4, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, sources=sources, target=target, is_wav=is_wav)

        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', is_wav=is_wav)

        self.max_duration = max_duration
        self.std = {}
        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            if set(self.sources) == set(__sources__):
                mixture = track.audio.transpose(1, 0)
            else:
                sources = []
                for _source in self.sources:
                    sources.append(track.targets[_source].audio.transpose(1, 0)[np.newaxis])
                sources = np.concatenate(sources, axis=0)
                mixture = sources.sum(axis=0)
            
            self.std[songID] = np.std(mixture.mean(axis=0))
            
            if max_duration is None:
                duration = track.duration
            else:
                if track.duration < max_duration:
                    duration = track.duration
                else:
                    duration = max_duration
            
            song_data = {
                'songID': songID,
                'patches': []
            }

            for start in np.arange(0, track.duration, duration):
                if start + duration > track.duration:
                    data = {
                        'start': start,
                        'duration': track.duration - start,
                        'padding_start': 0,
                        'padding_end': start + duration - track.duration
                    }
                else:
                    data = {
                        'start': start,
                        'duration': duration,
                        'padding_start': 0,
                        'padding_end': 0
                    }
                song_data['patches'].append(data)
            
            self.json_data.append(song_data)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: (batch_size, 1, T) if `target` is list, otherwise (batch_size, T)
            target <torch.Tensor>: (batch_size, len(target), T) if `target` is list, otherwise (batch_size, T)
            name <str>: Artist and title of song
        """
        song_data = self.json_data[idx]

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        name = track.name

        batch_mixture, batch_target = [], []
        max_samples = 0

        for data in song_data['patches']:
            track.chunk_start = data['start']
            track.chunk_duration = data['duration']

            if set(self.sources) == set(__sources__):
                mixture = track.audio.transpose(1, 0)
            else:
                sources = []
                for _source in self.sources:
                    sources.append(track.targets[_source].audio.transpose(1, 0)[np.newaxis])
                sources = np.concatenate(sources, axis=0)
                mixture = sources.sum(axis=0)
            
            if type(self.target) is list:
                target = []
                for _target in self.target:
                    target.append(track.targets[_target].audio.transpose(1, 0)[np.newaxis])
                target = np.concatenate(target, axis=0)
                mixture = mixture[np.newaxis]
            else:
                target = track.targets[self.target].audio.transpose(1, 0)

            mixture = torch.Tensor(mixture).float()
            target = torch.Tensor(target).float()

            max_samples = max(max_samples, mixture.size(-1))

            batch_mixture.append(mixture)
            batch_target.append(target)
        
        batch_mixture_padded, batch_target_padded = [], []
        start_segement = True

        for mixture, target in zip(batch_mixture, batch_target):
            if mixture.size(-1) < max_samples:
                padding = max_samples - mixture.size(-1)
                if start_segement:
                    mixture = F.pad(mixture, (padding, 0))
                    target = F.pad(target, (padding, 0))
                else:
                    mixture = F.pad(mixture, (0, padding))
                    target = F.pad(target, (0, padding))
            else:
                start_segement = False
            
            batch_mixture_padded.append(mixture.unsqueeze(dim=0))
            batch_target_padded.append(target.unsqueeze(dim=0))

        batch_mixture = torch.cat(batch_mixture_padded, dim=0) # batch_mixture : (batch_size, 1, n_mics, T) if `target` is list, otherwise (batch_size, n_mics, T)
        batch_target = torch.cat(batch_target_padded, dim=0) # batch_target : (batch_size, len(target), n_mics, T) if `target` is list, otherwise (batch_size, n_mics, T)

        n_dims = batch_mixture.dim()

        if n_dims == 3:
            # Use only first channel for validation
            batch_mixture, batch_target = batch_mixture[:, 0], batch_target[:, 0]
        elif n_dims == 4:
            batch_mixture, batch_target = batch_mixture[:, :, 0, :], batch_target[:, :, 0, :]
        else:
            raise ValueError("Invalid tensor shape. 2D or 3D tensor is expected, but givem {}D tensor".format(n_dims))
        
        return batch_mixture, batch_target, name
    
    def __len__(self):
        return len(self.json_data)

class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

        self.collate_fn = eval_collate_fn

def eval_collate_fn(batch):
    mixture, sources, name = batch[0]
    
    return mixture, sources, name