import random
import json

import numpy as np
import musdb
import torch
import torch.nn.functional as F

from dataset import WaveDataset, SpectrogramDataset

__sources__=['drums','bass','other','vocals']

EPS=1e-12
THRESHOLD_POWER=1e-5
MINSCALE = 0.75
MAXSCALE = 1.25

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, duration=4, overlap=None, sources=__sources__, target=None, json_path=None, augmentation=True, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train')

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return
        
        self.threshold = threshold
        self.duration = duration

        if overlap is None:
            overlap = self.duration / 2

        self.augmentation = augmentation

        self.json_data = {
            source: [] for source in sources
        }

        for songID, track in enumerate(self.mus.tracks):
            for start in np.arange(0, track.duration, duration - overlap):
                if start + duration >= track.duration:
                    break
                data = {
                    'songID': songID,
                    'start': start,
                    'duration': duration
                }
                for source in sources:
                    self.json_data[source].append(data)

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
        """
        if self.augmentation:
            mixture, target = self._getitem_augmentation()
        else:
            mixture, target = self._getitem(idx)

        return mixture, target
    
    def __len__(self):
        source = self.sources[0]
        
        return len(self.json_data[source])
    
    def _getitem(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
            title <str>: Title of song
        """
        _source = self.sources[0]

        data = self.json_data[_source][idx]

        songID = data['songID']
        track = self.mus.tracks[songID]
        
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

        return mixture, target

    def _getitem_augmentation(self):
        n_songs = len(self.mus.tracks)
        song_indices = random.choices(range(n_songs), k=len(self.sources))

        sources = []
        songIDs = []
        starts = []
        flips = []
        scales = []

        for _source, songID in zip(self.sources, song_indices):
            track = self.mus.tracks[songID]

            start = random.uniform(0, track.duration - self.duration)
            flip = random.choice([True, False])
            scale = random.uniform(MINSCALE, MAXSCALE)

            track.chunk_start = start
            track.chunk_duration = self.duration

            source = track.targets[_source].audio.transpose(1, 0)

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
                
                track = self.mus.tracks[songID]

                track.chunk_start = start
                track.chunk_duration = self.duration

                _target = track.targets[_target].audio.transpose(1, 0)
                
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
            
            track = self.mus.tracks[songID]
            
            track.chunk_start = start
            track.chunk_duration = self.duration

            _target = track.targets[_target].audio.transpose(1, 0)
                
            if flip:
                _target = _target[::-1]
            
            target = scale * _target

        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        return mixture, target
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, duration=10, overlap=None, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid')

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return

        self.duration = duration
        
        if overlap is None:
            overlap = self.duration / 2

        self.overlap = overlap
        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }
            for start in np.arange(-(duration - overlap), - duration, -(duration - overlap)):
                data = {
                    'start': 0,
                    'duration': duration + start,
                    'padding_start': -start,
                    'padding_end': 0
                }
                song_data['patches'].append(data)
            for start in np.arange(0, track.duration, duration - overlap):
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
        Args:
            idx <int>: index
        Returns:
            batch_mixture <torch.Tensor>: (n_segments, 1, 2, T_segment) if `target` is list, otherwise (n_segments, 2, T_segment)
            batch_target <torch.Tensor>: (n_segments, len(target), 2, T_segment) if `target` is list, otherwise (n_segments, 2, T_segment)
            T <int>: Length in time domain
            title <str>: Title of song
        """
        song_data = self.json_data[idx]

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        title = track.title

        T = track.duration

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

        batch_mixture = torch.cat(batch_mixture_padded, dim=0)
        batch_target = torch.cat(batch_target_padded, dim=0)
        
        return batch_mixture, batch_target, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class WaveTestDataset(WaveEvalDataset):
    def __init__(self, musdb18_root, sr=44100, duration=10, overlap=None, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="test")

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return

        self.duration = duration
        
        if overlap is None:
            overlap = self.duration / 2

        self.overlap = overlap
        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }
            for start in np.arange(-(duration - overlap), - duration, -(duration - overlap)):
                data = {
                    'start': 0,
                    'duration': duration + start,
                    'padding_start': -start,
                    'padding_end': 0
                }
                song_data['patches'].append(data)
            for start in np.arange(0, track.duration, duration - overlap):
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

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, patch_duration=4, overlap=None, sources=__sources__, target=None, json_path=None, augmentation=True, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train')

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return
        
        self.threshold = threshold
        self.patch_duration = patch_duration

        if overlap is None:
            overlap = self.patch_duration / 2

        self.augmentation = augmentation

        self.json_data = {
            source: [] for source in sources
        }

        for songID, track in enumerate(self.mus.tracks):
            for start in np.arange(0, track.duration, patch_duration - overlap):
                if start + patch_duration >= track.duration:
                    break
                data = {
                    'songID': songID,
                    'start': start,
                    'duration': patch_duration
                }
                for source in sources:
                    self.json_data[source].append(data)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
        """
        if self.augmentation:
            mixture, target = self._getitem_augmentation()
        else:
            mixture, target = self._getitem(idx)
        
        n_dims = mixture.dim()

        if n_dims > 2:
            mixture_channels = mixture.size()[:-1]
            target_channels = target.size()[:-1]
            mixture = mixture.reshape(-1, mixture.size(-1))
            target = target.reshape(-1, target.size(-1))

        mixture = torch.stft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, 2, n_bins, n_frames) or (2, n_bins, n_frames)
        target = torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), 2, n_bins, n_frames) or (2, n_bins, n_frames)
        
        if n_dims > 2:
            mixture = mixture.reshape(*mixture_channels, *mixture.size()[-2:])
            target = target.reshape(*target_channels, *target.size()[-2:])

        return mixture, target
    
    def __len__(self):
        source = self.sources[0]
        
        return len(self.json_data[source])
    
    def _getitem(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
            title <str>: Title of song
        """
        _source = self.sources[0]

        data = self.json_data[_source][idx]

        songID = data['songID']
        track = self.mus.tracks[songID]
        
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

        return mixture, target
    
    def _getitem_augmentation(self):
        n_songs = len(self.mus.tracks)
        song_indices = random.choices(range(n_songs), k=len(self.sources))

        sources = []
        songIDs = []
        starts = []
        flips = []
        scales = []

        for _source, songID in zip(self.sources, song_indices):
            track = self.mus.tracks[songID]

            start = random.uniform(0, track.duration - self.patch_duration)
            flip = random.choice([True, False])
            scale = random.uniform(MINSCALE, MAXSCALE)

            track.chunk_start = start
            track.chunk_duration = self.patch_duration

            source = track.targets[_source].audio.transpose(1, 0)

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
                
                track = self.mus.tracks[songID]
                track.chunk_start = start
                track.chunk_duration = self.patch_duration

                _target = track.targets[_target].audio.transpose(1, 0)
                
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
            
            track = self.mus.tracks[songID]
            track.chunk_start = start
            track.chunk_duration = self.patch_duration

            _target = track.targets[_target].audio.transpose(1, 0)
                
            if flip:
                _target = _target[::-1]
            
            target = scale * _target

        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        return mixture, target

    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, patch_duration=10, overlap=None, max_duration=None, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid')

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return

        self.patch_duration = patch_duration

        if max_duration is None:
            max_duration = patch_duration
        self.max_duration = patch_duration
        
        if overlap is None:
            overlap = self.patch_duration / 2

        self.overlap = overlap
        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }
            for start in np.arange(-(patch_duration - overlap), - patch_duration, -(patch_duration - overlap)):
                data = {
                    'start': 0,
                    'duration': patch_duration + start,
                    'padding_start': -start,
                    'padding_end': 0
                }
                song_data['patches'].append(data)
            
            max_duration = min(track.duration, self.max_duration)

            for start in np.arange(0, max_duration, patch_duration - overlap):
                if start + patch_duration > max_duration:
                    data = {
                        'start': start,
                        'duration': max_duration - start,
                        'padding_start': 0,
                        'padding_end': start + patch_duration - max_duration
                    }
                else:
                    data = {
                        'start': start,
                        'duration': patch_duration,
                        'padding_start': 0,
                        'padding_end': 0
                    }
                song_data['patches'].append(data)
            
            self.json_data.append(song_data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            title <str>: Title of song
        """
        song_data = self.json_data[idx]

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        title = track.title
        T = track.duration

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

        batch_mixture = torch.cat(batch_mixture_padded, dim=0)
        batch_target = torch.cat(batch_target_padded, dim=0)

        n_dims = batch_mixture.dim()

        if n_dims > 2:
            mixture_channels = batch_mixture.size()[:-1]
            target_channels = batch_target.size()[:-1]
            batch_mixture = batch_mixture.reshape(-1, batch_mixture.size(-1))
            batch_target = batch_target.reshape(-1, batch_target.size(-1))

        batch_mixture = torch.stft(batch_mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, 2, n_bins, n_frames) or (2, n_bins, n_frames)
        batch_target = torch.stft(batch_target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), 2, n_bins, n_frames) or (2, n_bins, n_frames)
        
        if n_dims > 2:
            batch_mixture = batch_mixture.reshape(*mixture_channels, *batch_mixture.size()[-2:])
            batch_target = batch_target.reshape(*target_channels, *batch_target.size()[-2:])
        
        return batch_mixture, batch_target, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class SpectrogramTestDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, patch_duration=5, max_duration=10, overlap=None, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="test")

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return

        self.patch_duration = patch_duration
        
        if overlap is None:
            overlap = self.patch_duration / 2

        self.overlap = overlap
        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }
            for start in np.arange(-(patch_duration - overlap), - patch_duration, -(patch_duration - overlap)):
                data = {
                    'start': 0,
                    'duration': patch_duration + start,
                    'padding_start': -start,
                    'padding_end': 0
                }
                song_data['patches'].append(data)
            for start in np.arange(0, track.duration, patch_duration - overlap):
                if start + patch_duration > track.duration:
                    data = {
                        'start': start,
                        'duration': track.duration - start,
                        'padding_start': 0,
                        'padding_end': start + patch_duration - track.duration
                    }
                else:
                    data = {
                        'start': start,
                        'duration': patch_duration,
                        'padding_start': 0,
                        'padding_end': 0
                    }
                song_data['patches'].append(data)
            self.json_data.append(song_data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            title <str>: Title of song
        """
        song_data = self.json_data[idx]

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        title = track.title
        T = track.duration

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

        batch_mixture = torch.cat(batch_mixture_padded, dim=0)
        batch_target = torch.cat(batch_target_padded, dim=0)

        n_dims = batch_mixture.dim()

        if n_dims > 2:
            mixture_channels = batch_mixture.size()[:-1]
            target_channels = batch_target.size()[:-1]
            batch_mixture = batch_mixture.reshape(-1, batch_mixture.size(-1))
            batch_target = batch_target.reshape(-1, batch_target.size(-1))

        batch_mixture = torch.stft(batch_mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, 2, n_bins, n_frames) or (2, n_bins, n_frames)
        batch_target = torch.stft(batch_target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), 2, n_bins, n_frames) or (2, n_bins, n_frames)
        
        if n_dims > 2:
            batch_mixture = batch_mixture.reshape(*mixture_channels, *batch_mixture.size()[-2:])
            batch_target = batch_target.reshape(*target_channels, *batch_target.size()[-2:])
        
        return batch_mixture, batch_target, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

"""
Data loader
"""
class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

        self.collate_fn = eval_collate_fn

def eval_collate_fn(batch):
    mixture, sources, T, title = batch[0]
    
    return mixture, sources, T, title

def _test_train_dataset():
    torch.manual_seed(111)
    
    musdb18_root = "../../../../../db/musdb18"

    dataset = SpectrogramTrainDataset(musdb18_root, fft_size=2048, hop_size=512, sr=8000, duration=4, target='vocals')
    
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

    dataset.save_as_json('data/tmp.json')

    dataset = SpectrogramTrainDataset.from_json(musdb18_root, 'data/tmp.json', fft_size=2048, hop_size=512, sr=44100, target='vocals')
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break


if __name__ == '__main__':
    _test_train_dataset()