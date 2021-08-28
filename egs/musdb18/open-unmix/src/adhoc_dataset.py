import random

import numpy as np
import musdb
import torch
import torch.nn.functional as F

from dataset import SpectrogramDataset

__sources__ = ['drums', 'bass', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12
THRESHOLD_POWER = 1e-5
MINSCALE = 0.75
MAXSCALE = 1.25

class SpectrogramTrainDataset(SpectrogramDataset):
    """
    Training dataset that returns randomly selected mixture spectrograms.
    In accordane with "D3Net: Densely connected multidilated DenseNet for music source separation," training dataset includes all 100 songs.
    """
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=SAMPLE_RATE_MUSDB18, patch_samples=4*SAMPLE_RATE_MUSDB18, overlap=None, samples_per_epoch=None, sources=__sources__, target=None, augmentation=True, threshold=THRESHOLD_POWER, is_wav=False):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target, is_wav=is_wav)
        
        assert_sample_rate(sr)
        self.sr = sr
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", is_wav=is_wav) # train (86 songs) + valid (14 songs)
        
        self.threshold = threshold
        self.patch_samples = patch_samples

        self.augmentation = augmentation

        if augmentation:
            if samples_per_epoch is None:
                patch_duration = patch_samples / sr
                total_duration = 0
                for track in self.mus.tracks:
                    total_duration += track.duration
                samples_per_epoch = int(total_duration / patch_duration) # 3862 is expected.

            self.samples_per_epoch = samples_per_epoch
            self.json_data = None
        else:
            if overlap is None:
                overlap = self.patch_samples / 2
            
            self.samples_per_epoch = None
            self.json_data = {
                source: [] for source in sources
            }

            for songID, track in enumerate(self.mus.tracks):
                samples = track.audio.shape[0]
                for start in np.arange(0, samples, patch_samples - overlap):
                    if start + patch_samples >= samples:
                        break
                    data = {
                        'songID': songID,
                        'start': start / sr,
                        'duration': patch_samples / sr
                    }
                    for source in sources:
                        self.json_data[source].append(data)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
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

        mixture = torch.stft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        target = torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        
        if n_dims > 2:
            mixture = mixture.reshape(*mixture_channels, *mixture.size()[-2:])
            target = target.reshape(*target_channels, *target.size()[-2:])

        return mixture, target
    
    def __len__(self):
        if self.augmentation:
            return self.samples_per_epoch
        else:
            source = self.sources[0]
            
            return len(self.json_data[source])
    
    def _getitem(self, idx):
        """
        Returns time domain signals
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
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

        mixture, target = torch.from_numpy(mixture).float(), torch.from_numpy(target).float()

        track.chunk_start = 0
        track.chunk_duration = None

        return mixture, target
    
    def _getitem_augmentation(self):
        """
        Returns time domain signals
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
            name <str>: Artist and title of song
        """
        n_songs = len(self.mus.tracks)
        song_indices = random.choices(range(n_songs), k=len(self.sources))

        sources = []

        for _source, songID in zip(self.sources, song_indices):
            track = self.mus.tracks[songID]

            start = random.uniform(0, track.duration - self.patch_samples / self.sr)
            flip = random.choice([True, False])
            scale = random.uniform(MINSCALE, MAXSCALE)

            track.chunk_start = start
            track.chunk_duration = self.patch_samples / self.sr

            source = track.targets[_source].audio.transpose(1, 0)

            if flip:
                source = source[::-1]

            sources.append(scale * source[np.newaxis])
        
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

        mixture, target = torch.from_numpy(mixture).float(), torch.from_numpy(target).float()

        return mixture, target

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=SAMPLE_RATE_MUSDB18, samples=10*SAMPLE_RATE_MUSDB18, max_samples=None, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target, is_wav=is_wav)
        
        assert_sample_rate(sr)
        self.sr = sr
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', is_wav=is_wav)

        self.samples = samples

        if max_samples is None:
            max_samples = samples
        
        self.max_samples = max_samples

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }

            samples = track.audio.shape[0]
            
            max_samples = min(samples, self.max_samples)

            song_data['start'] = 0
            song_data['samples'] = max_samples

            self.json_data.append(song_data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            name <str>: Artist and title of song
        """
        song_data = self.json_data[idx]

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        name = track.name

        audio = {
            'mixture': track.audio.transpose(1, 0)
        }

        for _source in self.sources:
            audio[_source] = track.targets[_source].audio.transpose(1, 0)

        start = song_data['start']
        end = start + song_data['samples']

        if set(self.sources) == set(__sources__):
            mixture = audio['mixture'][:, start: end]
        else:
            sources = []
            for _source in self.sources:
                sources.append(audio[_source][np.newaxis, :, start: end])
            sources = np.concatenate(sources, axis=0)
            mixture = sources.sum(axis=0)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                target.append(audio[_target][np.newaxis, :, start: end])
            target = np.concatenate(target, axis=0)
            mixture = mixture[np.newaxis]
        else:
            target = audio[self.target][:, start: end]

        mixture, target = torch.from_numpy(mixture).float(), torch.from_numpy(target).float()

        n_dims = mixture.dim()

        if n_dims > 2:
            mixture_channels = mixture.size()[:-1]
            target_channels = target.size()[:-1]
            mixture = mixture.reshape(-1, mixture.size(-1))
            target = target.reshape(-1, target.size(-1))

        mixture = torch.stft(mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        target = torch.stft(target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        
        if n_dims > 2:
            mixture = mixture.reshape(*mixture_channels, *mixture.size()[-2:])
            target = target.reshape(*target_channels, *target.size()[-2:])
        
        return mixture, target, name

"""
Data loader
"""
class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

def assert_sample_rate(sr):
    assert sr == SAMPLE_RATE_MUSDB18, "sample rate is expected {}, but given {}".format(SAMPLE_RATE_MUSDB18, sr)