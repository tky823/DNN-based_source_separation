import random

import numpy as np
import musdb
import torch

from dataset import MUSDB18Dataset

__sources__ = ['drums', 'bass', 'other', 'vocals']

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
        super().__init__(musdb18_root, sr=SAMPLE_RATE_MUSDB18, sources=sources, target=target, is_wav=is_wav)

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
        return 0

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
            mixture <torch.Tensor>: (1, 1, T) if `target` is list, otherwise (1, T)
            target <torch.Tensor>: (len(target), 1, T) if `target` is list, otherwise (1, T)
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

            sources.append(scale * source[np.newaxis, np.newaxis])
            songIDs.append(songID)
            starts.append(start)
            channels.append(channel)
            scales.append(scale)
        
        sources = np.concatenate(sources, axis=0)
        mixture = sources.sum(axis=0)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                idx = self.sources.index(_target)
                songID = songIDs[idx]
                start = starts[idx]
                channel = channels[idx]
                scale = scales[idx]
                std = self.std[songID]
                
                track = self.mus.tracks[songID]

                track.chunk_start = start
                track.chunk_duration = self.duration

                _target = track.targets[_target].audio.transpose(1, 0) / std
                _target = _target[channel]

                target.append(scale * _target[np.newaxis, np.newaxis])
            
            target = np.concatenate(target, axis=0)
            mixture = mixture[np.newaxis]
        else:
            _target = self.target
            idx = self.sources.index(_target)
            songID = songIDs[idx]
            start = starts[idx]
            channel = channels[idx]
            scale = scales[idx]
            std = self.std[songID]
            
            track = self.mus.tracks[songID]
            
            track.chunk_start = start
            track.chunk_duration = self.duration

            _target = track.targets[_target].audio.transpose(1, 0) / std
            _target = _target[channel]
            
            target = scale * _target[np.newaxis]

        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        return mixture, target

    def __len__(self):
        return self.samples_per_epoch

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, max_duration=4, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, sources=sources, target=target, is_wav=is_wav)

        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', is_wav=is_wav)

        self.max_duration = max_duration

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            if max_duration is None:
                duration = track.duration
            else:
                if track.duration < max_duration:
                    duration = track.duration
                else:
                    duration = max_duration
            
            data = {
                'songID': songID,
                'start': 0,
                'duration': duration
            }
            self.json_data.append(data)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: (1, 1, T) if `target` is list, otherwise (1, T)
            target <torch.Tensor>: (len(target), 1, T) if `target` is list, otherwise (1, T)
            name <str>: Artist and title of song
        """
        mixture, target, name = super().__getitem__(idx) # mixture
        # mixture : (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
        # target : (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
        
        n_dims = mixture.dim()

        if n_dims == 2:
            # Use only first channel for validation
            mixture, target = mixture[0], target[0]
        elif n_dims == 3:
            mixture, target = mixture[:, 0, :], target[:, 0, :]
        else:
            raise ValueError("Invalid tensor shape. 2D or 3D tensor is expected, but givem {}D tensor".format(n_dims))

        return mixture, target, name