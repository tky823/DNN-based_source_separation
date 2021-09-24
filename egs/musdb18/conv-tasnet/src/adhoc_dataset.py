import os
import random

import torch
import torchaudio

from dataset import WaveDataset

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, duration=4, overlap=None, samples_per_epoch=None, sources=__sources__, target=None, augmentation=None):
        """
        Args:
            musdb18_root <int>: Path to MUSDB or MUSDB-HQ
            sr: Sampling frequency. Default: 44100
            sources <list<str>>: Sources included in mixture
            target <str> or <list<str>>: Target source(s)
        """
        super().__init__(
            musdb18_root,
            sr=SAMPLE_RATE_MUSDB18, # WaveDataset's sr is expected SAMPLE_RATE_MUSDB18
            sources=sources,
            target=target
        )

        train_txt_path = os.path.join(musdb18_root, 'train.txt')

        with open(train_txt_path, 'r') as f:
            names = [line.strip() for line in f]

        self.sr = sr
        self.samples = int(duration * sr)
        self.augmentation = augmentation

        self.tracks = []

        if augmentation:
            total_duration = 0

            for trackID, name in enumerate(names):
                mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
                audio_info = torchaudio.info(mixture_path)
                track_sr = audio_info.sample_rate
                track_samples = audio_info.num_frames

                track = {
                    'name': name,
                    'samples_original': track_samples,
                    'path': {
                        'mixture': mixture_path
                    }
                }
                
                for source in sources:
                    track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))
                
                self.tracks.append(track)

                track_duration = track_samples / track_sr
                total_duration += track_duration

            if samples_per_epoch is None:
                samples_per_epoch = int(total_duration / duration)

            self.samples_per_epoch = samples_per_epoch
            self.json_data = None
        else:
            samples_original = int(self.samples * SAMPLE_RATE_MUSDB18 / sr)

            if overlap is None:
                overlap = samples_original // 2
            self.samples_per_epoch = None

            for trackID, name in enumerate(names):
                mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
                audio_info = torchaudio.info(mixture_path)
                track_sr = audio_info.sample_rate
                track_samples = audio_info.num_frames

                track = {
                    'name': name,
                    'samples_original': track_samples,
                    'path': {
                        'mixture': mixture_path
                    }
                }

                for source in sources:
                    track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))
                self.tracks.append(track)

                for start in range(0, track_samples, self.samples - overlap):
                    if start + self.samples >= track_samples:
                        break
                    data = {
                        'trackID': trackID,
                        'start': start,
                        'samples_original': samples_original,
                    }
                    self.json_data.append(data)

        if sr != SAMPLE_RATE_MUSDB18:
            self.pre_resampler = torchaudio.transforms.Resample(SAMPLE_RATE_MUSDB18, sr)
        else:
            self.pre_resampler = None

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Tensor with shape (1, n_mics, T)  if `target` is list, otherwise (n_mics, T) 
            target <torch.Tensor>: Tensor with shape (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
        """
        if self.augmentation:
            mixture, target = self._getitem_augmentation()
        else:
            raise NotImplementedError("Implement _getitem()")
        
        if self.pre_resampler is not None:
            mixture_channels, target_channels = mixture.size()[:-1], target.size()[:-1]
            mixture, target = mixture.reshape(-1, mixture.size(-1)), target.reshape(-1, target.size(-1))

            mixture = self.pre_resampler(mixture)
            target = self.pre_resampler(target)

            mixture, target = mixture.reshape(*mixture_channels, mixture.size(-1)), target.reshape(*target_channels, target.size(-1))

        return mixture, target

    def _getitem_augmentation(self):
        """
        Returns time domain signals
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
            name <str>: Artist and title of track
        """
        n_tracks = len(self.tracks)
        track_indices = random.choices(range(n_tracks), k=len(self.sources))

        sources = []

        for _source, trackID in zip(self.sources, track_indices):
            track = self.tracks[trackID]
            source_path = track['path'][_source]
            track_samples = track['samples_original']

            start = random.randint(0, track_samples - self.samples - 1)
            source, _ = torchaudio.load(source_path, frame_offset=start, num_frames=self.samples)

            # Apply augmentation
            source = self.augmentation(source)
            sources.append(source.unsqueeze(dim=0))
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                source_idx = self.sources.index(_target)
                _target = sources[source_idx]
                target.append(_target)
            target = torch.cat(target, dim=0)

            sources = torch.cat(sources, dim=0)
            mixture = sources.sum(dim=0, keepdim=True)
        else:
            source_idx = self.sources.index(self.target)
            target = sources[source_idx]
            target = target.squeeze(dim=0)

            sources = torch.cat(sources, dim=0)
            mixture = sources.sum(dim=0)

        return mixture, target

    def __len__(self):
        return self.samples_per_epoch

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, max_duration=60, sources=__sources__, target=None):
        super().__init__(
            musdb18_root,
            sr=SAMPLE_RATE_MUSDB18, # WaveDataset's sr is expected SAMPLE_RATE_MUSDB18
            sources=sources,
            target=target
        )

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        
        with open(valid_txt_path, 'r') as f:
            names = [line.strip() for line in f]

        self.max_samples = int(sr * max_duration)

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
            audio_info = torchaudio.info(mixture_path)
            track_sr = audio_info.sample_rate
            track_samples = audio_info.num_frames
            samples_original = min(int(self.max_samples * track_sr / sr), track_samples)

            track = {
                'name': name,
                'samples_original': track_samples,
                'path': {
                    'mixture': mixture_path
                }
            }

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))
            
            track_data = {
                'trackID': trackID,
                'start': 0,
                'samples_original': samples_original
            }
            
            self.tracks.append(track)
            self.json_data.append(track_data) # len(self.json_data) determines # of samples in dataset
        
        if sr != SAMPLE_RATE_MUSDB18:
            self.pre_resampler = torchaudio.transforms.Resample(SAMPLE_RATE_MUSDB18, sr)
        else:
            self.pre_resampler = None

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, T)  if `target` is list, otherwise (n_mics, T) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
            name <str>: Artist and title of track
        """
        track_data = self.json_data[idx]

        trackID = track_data['trackID']
        track = self.tracks[trackID]
        name = track['name']
        paths = track['path']
        samples_original = track_data['samples_original']

        if set(self.sources) == set(__sources__):
            mixture, _ = torchaudio.load(paths['mixture'], num_frames=samples_original) # (n_mics, T)
        else:
            sources = []
            for _source in self.sources:
                source, _ = torchaudio.load(paths[_source], num_frames=samples_original) # (n_mics, T)
                sources.append(source.unsqueeze(dim=0))
            sources = torch.cat(sources, dim=0) # (len(self.sources), n_mics, T)
            mixture = sources.sum(dim=0) # (n_mics, T)
        
        if type(self.target) is list:
            target = []
            for _target in self.target:
                source, _ = torchaudio.load(paths[_target], num_frames=samples_original) # (n_mics, T)
                target.append(source.unsqueeze(dim=0))
            target = torch.cat(target, dim=0) # (len(target), n_mics, T)
            mixture = mixture.unsqueeze(dim=0) # (1, n_mics, T)
        else:
            # mixture: (n_mics, T)
            target, _ = torchaudio.load(paths[self.target], num_frames=samples_original) # (n_mics, T)
        
        if self.pre_resampler is not None:
            mixture_channels, target_channels = mixture.size()[:-1], target.size()[:-1]
            mixture, target = mixture.reshape(-1, mixture.size(-1)), target.reshape(-1, target.size(-1))

            mixture = self.pre_resampler(mixture)
            target = self.pre_resampler(target)

            mixture, target = mixture.reshape(*mixture_channels, mixture.size(-1)), target.reshape(*target_channels, target.size(-1))

        return mixture, target, name

def assert_sample_rate(sr):
    assert sr == SAMPLE_RATE_MUSDB18, "sample rate is expected {}, but given {}".format(SAMPLE_RATE_MUSDB18, sr)