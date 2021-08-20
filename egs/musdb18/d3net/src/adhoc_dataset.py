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
                        'start': start,
                        'duration': patch_samples
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
            name <str>: Artist and title of song
        """
        raise NotImplementedError("Not support naive __getitem__")
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
        songIDs = []
        starts = []
        flips = []
        scales = []

        for _source, songID in zip(self.sources, song_indices):
            track = self.mus.tracks[songID]
            samples = track.audio.shape[0]

            start = random.randint(0, samples - self.patch_samples - 1)
            flip = random.choice([True, False])
            scale = random.uniform(MINSCALE, MAXSCALE)
            end = start + self.patch_samples

            source = track.targets[_source].audio.transpose(1, 0)[:, start: end]

            if flip:
                source = source[::-1]

            sources.append(scale * source[np.newaxis])
            songIDs.append(songID)
            starts.append(start)
            flips.append(flip)
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

        mixture, target = torch.from_numpy(mixture).float(), torch.from_numpy(target).float()

        return mixture, target

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=SAMPLE_RATE_MUSDB18, patch_samples=10*SAMPLE_RATE_MUSDB18, max_samples=None, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target, is_wav=is_wav)
        
        assert_sample_rate(sr)
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', is_wav=is_wav)

        self.patch_samples = patch_samples

        if max_samples is None:
            max_samples = patch_samples
        
        self.max_samples = max_samples

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }

            samples = track.audio.shape[0]
            
            max_samples = min(samples, self.max_samples)

            for start in np.arange(0, max_samples, patch_samples):
                if start + patch_samples > max_samples:
                    data = {
                        'start': start,
                        'samples': max_samples - start,
                        'padding_start': 0,
                        'padding_end': start + patch_samples - max_samples
                    }
                else:
                    data = {
                        'start': start,
                        'samples': patch_samples,
                        'padding_start': 0,
                        'padding_end': 0
                    }
                song_data['patches'].append(data)
            
            self.json_data.append(song_data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            T <float>: Duration [sec]
            name <str>: Artist and title of song
        """
        song_data = self.json_data[idx]

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        name = track.name

        batch_mixture, batch_target = [], []
        max_samples = 0

        for data in song_data['patches']:
            start = data['start']
            end = start + data['samples']

            if set(self.sources) == set(__sources__):
                mixture = track.audio.transpose(1, 0)[:, start: end]
            else:
                sources = []
                for _source in self.sources:
                    sources.append(track.targets[_source].audio.transpose(1, 0)[np.newaxis, :, start: end])
                sources = np.concatenate(sources, axis=0)
                mixture = sources.sum(axis=0)
            
            if type(self.target) is list:
                target = []
                for _target in self.target:
                    target.append(track.targets[_target].audio.transpose(1, 0)[np.newaxis, :, start: end])
                target = np.concatenate(target, axis=0)
                mixture = mixture[np.newaxis]
            else:
                target = track.targets[self.target].audio.transpose(1, 0)[:, start: end]

            mixture, target = torch.from_numpy(mixture).float(), torch.from_numpy(target).float()

            max_samples = max(max_samples, mixture.size(-1))

            batch_mixture.append(mixture)
            batch_target.append(target)
        
        batch_mixture_padded, batch_target_padded = [], []
        start_segement = True

        # TODO: remove start segement
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

        batch_mixture = torch.stft(batch_mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        batch_target = torch.stft(batch_target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        
        if n_dims > 2:
            batch_mixture = batch_mixture.reshape(*mixture_channels, *batch_mixture.size()[-2:])
            batch_target = batch_target.reshape(*target_channels, *batch_target.size()[-2:])
        
        return batch_mixture, batch_target, name

class SpectrogramTestDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=SAMPLE_RATE_MUSDB18, patch_samples=5*SAMPLE_RATE_MUSDB18, sources=__sources__, target=None, is_wav=False):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target, is_wav=is_wav)
        
        assert_sample_rate(sr)
        self.mus = musdb.DB(root=self.musdb18_root, subsets="test", is_wav=is_wav)

        self.patch_samples = patch_samples
        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            song_data = {
                'songID': songID,
                'patches': []
            }
            samples = track.audio.shape[0]
            
            for start in np.arange(0, samples, patch_samples):
                if start + patch_samples > samples:
                    data = {
                        'start': start,
                        'samples': samples - start,
                        'padding_start': 0,
                        'padding_end': start + patch_samples - samples
                    }
                else:
                    data = {
                        'start': start,
                        'samples': patch_samples,
                        'padding_start': 0,
                        'padding_end': 0
                    }
                song_data['patches'].append(data)
            self.json_data.append(song_data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            samples <int>: Number of samples in time domain.
            name <str>: Artist and title of song
        """
        song_data = self.json_data[idx]
        patch_samples = self.patch_samples

        songID = song_data['songID']
        track = self.mus.tracks[songID]
        name = track.name
        samples, _ = track.audio.shape

        if set(self.sources) == set(__sources__):
            mixture = track.audio.transpose(1, 0) # (n_mics, T)
        else:
            sources = []
            for _source in self.sources:
                sources.append(track.targets[_source].audio.transpose(1, 0)[np.newaxis])
            sources = np.concatenate(sources, axis=0) # (n_mics, T)
            mixture = sources.sum(axis=0) # (n_mics, T)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                target.append(track.targets[_target].audio.transpose(1, 0)[np.newaxis])
            target = np.concatenate(target, axis=0) # (len(target), n_mics, T)
            mixture = mixture[np.newaxis] # (1, n_mics, T)
        else:
            # mixture: (n_mics, T)
            target = track.targets[self.target].audio.transpose(1, 0) # (n_mics, T)

        mixture, target = torch.from_numpy(mixture).float(), torch.from_numpy(target).float()

        padding = (patch_samples - samples % patch_samples) % patch_samples

        mixture_padded, target_padded = F.pad(mixture, (0, padding)), F.pad(target, (0, padding))
        mixture_padded, target_padded = mixture_padded.reshape(*mixture_padded.size()[:-1], -1, patch_samples), target_padded.reshape(*target_padded.size()[:-1], -1, patch_samples)

        if type(self.target) is list:
            # mixture_padded: (1, n_mics, batch_size, patch_samples), target_padded: (len(target), n_mics, batch_size, patch_samples)
            batch_mixture, batch_target = mixture_padded.permute(2, 0, 1, 3).contiguous(), target_padded.permute(2, 0, 1, 3).contiguous()
            # batch_mixture: (batch_size, 1, n_mics, patch_samples), batch_target: (batch_size, len(target), n_mics, patch_samples)
        else:
            # mixture_padded: (n_mics, batch_size, patch_samples), target_padded: (n_mics, batch_size, patch_samples)
            batch_mixture, batch_target = mixture_padded.permute(1, 0, 2).contiguous(), target_padded.permute(1, 0, 2).contiguous()
            # batch_mixture: (batch_size, n_mics, patch_samples), batch_target: (batch_size, n_mics, patch_samples)

        n_dims = batch_mixture.dim()

        if n_dims > 2:
            mixture_channels = batch_mixture.size()[:-1]
            target_channels = batch_target.size()[:-1]
            batch_mixture = batch_mixture.reshape(-1, batch_mixture.size(-1))
            batch_target = batch_target.reshape(-1, batch_target.size(-1))

        batch_mixture = torch.stft(batch_mixture, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        batch_target = torch.stft(batch_target, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        
        if n_dims > 2:
            batch_mixture = batch_mixture.reshape(*mixture_channels, *batch_mixture.size()[-2:])
            batch_target = batch_target.reshape(*target_channels, *batch_target.size()[-2:])
        
        return batch_mixture, batch_target, samples, name

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

def assert_sample_rate(sr):
    assert sr == SAMPLE_RATE_MUSDB18, "sample rate is expected {}, but given {}".format(SAMPLE_RATE_MUSDB18, sr)