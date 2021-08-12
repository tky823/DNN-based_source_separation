import random

import numpy as np
import musdb
import torch

from dataset import WaveDataset

__sources__ = ['drums', 'bass', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12
THRESHOLD_POWER = 1e-5
MINSCALE = 0.75
MAXSCALE = 1.25

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, duration=4, samples_per_epoch=None, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target, is_wav=is_wav)
        
        assert_sample_rate(sr)
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train', is_wav=is_wav)
        
        self.duration = duration

        self.std = {}
        self.json_data = None

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
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
        """
        n_songs = len(self.mus.tracks)
        song_indices = random.choices(range(n_songs), k=len(self.sources))

        sources = []
        songIDs = []
        starts = []
        flips = []
        scales = []

        for _source, songID in zip(self.sources, song_indices):
            track = self.mus.tracks[songID]
            std = self.std[songID]

            start = random.uniform(0, track.duration - self.duration)
            flip = random.choice([True, False])
            scale = random.uniform(MINSCALE, MAXSCALE)

            track.chunk_start = start
            track.chunk_duration = self.duration

            source = track.targets[_source].audio.transpose(1, 0) / std

            if flip:
                source = source[::-1]

            sources.append(scale * source[np.newaxis])
            songIDs.append(songID)
            starts.append(start)
            flips.append(flip)
            scales.append(scale)
        
        sources = np.concatenate(sources, axis=0)
        mixture = sources.sum(axis=0)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                idx = self.sources.index(_target)
                songID = songIDs[idx]
                start = starts[idx]
                flip = flips[idx]
                scale = scales[idx]
                std = self.std[songID]
                
                track = self.mus.tracks[songID]

                track.chunk_start = start
                track.chunk_duration = self.duration

                _target = track.targets[_target].audio.transpose(1, 0) / std
                
                if flip:
                    _target = _target[::-1]

                target.append(scale * _target[np.newaxis])
            
            target = np.concatenate(target, axis=0)
            mixture = mixture[np.newaxis]
        else:
            _target = self.target
            idx = self.sources.index(_target)
            songID = songIDs[idx]
            start = starts[idx]
            flip = flips[idx]
            scale = scales[idx]
            std = self.std[songID]
            
            track = self.mus.tracks[songID]
            
            track.chunk_start = start
            track.chunk_duration = self.duration

            _target = track.targets[_target].audio.transpose(1, 0) / std
                
            if flip:
                _target = _target[::-1]
            
            target = scale * _target

        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        return mixture, target

    def __len__(self):
        return self.samples_per_epoch

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, max_duration=4, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target, is_wav=is_wav)

        assert_sample_rate(sr)
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

def assert_sample_rate(sr):
    assert sr == SAMPLE_RATE_MUSDB18, "sample rate is expected {}, but given {}".format(SAMPLE_RATE_MUSDB18, sr)