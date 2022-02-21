import os
import random

import torch
import torchaudio

from utils.audio import build_window
from transforms.stft import stft

__sources__ = ['bass', 'drums', 'other', 'vocals']

SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        """
        Args:
            musdb18_root <str>: Path to MUSDB18 root.
            sample_rate <int>: Sampling rate.
            sources <list<str>>: Sources for mixture. Default: ['bass', 'drums', 'other', 'vocals']
            target <str> or <list<str>>: Target source(s). If None is given, `sources` is used by default.
        """
        super().__init__()

        assert_sample_rate(sample_rate)

        if target is not None:
            if type(target) is list:
                for _target in target:
                    assert _target in sources, "`sources` doesn't contain target {}".format(_target)
            else:
                assert target in sources, "`sources` doesn't contain target {}".format(target)
        else:
            target = sources

        self.musdb18_root = os.path.abspath(musdb18_root)
        self.tracks = []

        self.sources = sources
        self.target = target

class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        """
        Args:
            musdb18_root <int>: Path to MUSDB or MUSDB-HQ
            sample_rate: Sampling frequency. Default: 44100
            sources <list<str>>: Sources included in mixture
            target <str> or <list<str>>: Target source(s)
        """
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)

        self.json_data = None

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
            name <str>: Artist and title of track
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
            target = []
            for _target in self.target:
                source, _ = torchaudio.load(paths[_target], frame_offset=start, num_frames=samples)
                target.append(source.unsqueeze(dim=0))
            target = torch.cat(target, dim=0)
            mixture = mixture.unsqueeze(dim=0)
        else:
            target, _ = torchaudio.load(paths[self.target], frame_offset=start, num_frames=samples)

        return mixture, target, name

    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, samples=4*SAMPLE_RATE_MUSDB18, overlap=None, sources=__sources__, target=None, include_valid=False):
        """
        Args:
            include_valid <bool>: Include validation data for training.
        """
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        train_txt_path = os.path.join(musdb18_root, 'train.txt')

        with open(valid_txt_path, 'r') as f:
            valid_lst = [line.strip() for line in f]

        names = []

        with open(train_txt_path, 'r') as f:
            for line in f:
                name = line.strip()

                if (not include_valid) and name in valid_lst:
                    continue

                names.append(name)

        if overlap is None:
            overlap = samples // 2

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
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
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))

            self.tracks.append(track)

            for start in range(0, track_samples, samples - overlap):
                if start + samples >= track_samples:
                    break
                data = {
                    'trackID': trackID,
                    'start': start,
                    'samples': samples,
                }
                self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
        """
        mixture, target, _ = super().__getitem__(idx)

        return mixture, target

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, max_samples=4*SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')

        names = []
        with open(valid_txt_path, 'r') as f:
            for line in f:
                name = line.strip()
                names.append(name)

        self.max_samples = max_samples

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
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

            if max_samples is None:
                samples = track_samples
            else:
                if track_samples < max_samples:
                    samples = track_samples
                else:
                    samples = max_samples

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))

            self.tracks.append(track)

            data = {
                'trackID': trackID,
                'start': 0,
                'samples': samples
            }

            self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: (1, n_mics, T) if `target` is list, otherwise (n_mics, T)
            target <torch.Tensor>: (len(target), n_mics, T) if `target` is list, otherwise (n_mics, T)
        """
        mixture, target, _ = super().__getitem__(idx)

        return mixture, target

class WaveTestDataset(WaveDataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        super().__init__(musdb18_root, sample_rate=sample_rate, sources=sources, target=target)

        test_txt_path = os.path.join(musdb18_root, 'test.txt')

        names = []
        with open(test_txt_path, 'r') as f:
            for line in f:
                name = line.strip()
                names.append(name)

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

            self.tracks.append(track)

            data = {
                'trackID': trackID,
                'start': 0,
                'samples': track_samples
            }

            self.json_data.append(data)

class SpectrogramDataset(WaveDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
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

    def _is_active(self, input, threshold=1e-5):
        n_dims = input.dim()

        if n_dims > 2:
            input = input.reshape(-1, input.size(-1))

        input = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources)*n_mics, n_bins, n_frames)
        power = torch.sum(torch.abs(input)**2, dim=-1) # (len(sources)*n_mics, n_bins, n_frames)
        power = torch.mean(power)

        if power.item() >= threshold:
            return True
        else:
            return False

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            name <str>: Artist and title of track
        """
        mixture, target, name = super().__getitem__(idx)

        T = mixture.size(-1)

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        target = stft(target, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)

        return mixture, target, T, name

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, samples=4*SAMPLE_RATE_MUSDB18, overlap=None, sources=__sources__, target=None, include_valid=False):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)

        assert_sample_rate(sample_rate)

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        train_txt_path = os.path.join(musdb18_root, 'train.txt')

        with open(valid_txt_path, 'r') as f:
            valid_lst = [line.strip() for line in f]

        names = []

        with open(train_txt_path, 'r') as f:
            for line in f:
                name = line.strip()

                if (not include_valid) and name in valid_lst:
                    continue

                names.append(name)

        if overlap is None:
            overlap = samples // 2

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
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
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))

            self.tracks.append(track)

            for start in range(0, track_samples, samples - overlap):
                if start + samples >= track_samples:
                    break
                data = {
                    'trackID': trackID,
                    'start': start,
                    'samples': samples,
                }
                self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
        """
        mixture, target, _, _ = super().__getitem__(idx)

        return mixture, target

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, max_samples=10*SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')

        with open(valid_txt_path, 'r') as f:
            names = [line.strip() for line in f]

        self.max_samples = max_samples

        self.tracks = []
        self.json_data = []

        for trackID, name in enumerate(names):
            mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
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

            if max_samples is None:
                samples = track_samples
            else:
                if track_samples < max_samples:
                    samples = track_samples
                else:
                    samples = max_samples

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))

            self.tracks.append(track)

            data = {
                'trackID': trackID,
                'start': 0,
                'samples': samples
            }

            self.json_data.append(data)

    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            name <str>: Artist and title of track
        """
        mixture, sources, T, name = super().__getitem__(idx)

        return mixture, sources, T, name

class SpectrogramTestDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, max_samples=10*SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)

        test_txt_path = os.path.join(musdb18_root, 'test.txt')

        names = []
        with open(test_txt_path, 'r') as f:
            for line in f:
                name = line.strip()
                names.append(name)

        self.max_samples = max_samples

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

            if max_samples is None:
                samples = track_samples
            else:
                if track_samples < max_samples:
                    samples = track_samples
                else:
                    samples = max_samples

            for source in sources:
                track['path'][source] = os.path.join(musdb18_root, 'test', name, "{}.wav".format(source))

            self.tracks.append(track)

            data = {
                'trackID': trackID,
                'start': 0,
                'samples': samples
            }

            self.json_data.append(data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            name <str>: Artist and title of track
        """
        mixture, sources, T, name = super().__getitem__(idx)

        return mixture, sources, T, name

"""
    Augmentation dataset
"""
class AugmentationWaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sample_rate=SAMPLE_RATE_MUSDB18, duration=4, overlap=None, samples_per_epoch=None, sources=__sources__, target=None, include_valid=False, augmentation=None):
        """
        Args:
            musdb18_root <int>: Path to MUSDB or MUSDB-HQ
            sample_rate: Sampling frequency. Default: 44100
            sources <list<str>>: Sources included in mixture
            target <str> or <list<str>>: Target source(s)
        """
        super().__init__(
            musdb18_root,
            sample_rate=SAMPLE_RATE_MUSDB18, # WaveDataset's sample_rate is expected SAMPLE_RATE_MUSDB18
            sources=sources,
            target=target
        )

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        train_txt_path = os.path.join(musdb18_root, 'train.txt')

        with open(valid_txt_path, 'r') as f:
            valid_lst = [line.strip() for line in f]

        names = []

        with open(train_txt_path, 'r') as f:
            for line in f:
                name = line.strip()

                if (not include_valid) and name in valid_lst:
                    continue

                names.append(name)

        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate)
        self.augmentation = augmentation

        self.tracks = []

        if augmentation:
            total_duration = 0

            for trackID, name in enumerate(names):
                mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
                audio_info = torchaudio.info(mixture_path)
                track_sample_rate = audio_info.sample_rate
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

                track_duration = track_samples / track_sample_rate
                total_duration += track_duration

            if samples_per_epoch is None:
                samples_per_epoch = int(total_duration / duration)

            self.samples_per_epoch = samples_per_epoch
            self.json_data = None
        else:
            samples_original = int(self.samples * SAMPLE_RATE_MUSDB18 / sample_rate)

            if overlap is None:
                overlap = samples_original // 2
            self.samples_per_epoch = None

            for trackID, name in enumerate(names):
                mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
                audio_info = torchaudio.info(mixture_path)
                track_sample_rate = audio_info.sample_rate
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

        if sample_rate != SAMPLE_RATE_MUSDB18:
            self.pre_resampler = torchaudio.transforms.Resample(SAMPLE_RATE_MUSDB18, sample_rate)
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

class AugmentationSpectrogramTrainDataset(SpectrogramDataset):
    """
    Training dataset that returns randomly selected mixture spectrograms.
    """
    def __init__(self, musdb18_root, n_fft, hop_length=None, window_fn='hann', normalize=False, sample_rate=SAMPLE_RATE_MUSDB18, patch_samples=6*SAMPLE_RATE_MUSDB18, overlap=None, samples_per_epoch=None, sources=__sources__, target=None, include_valid=False, augmentation=None):
        super().__init__(musdb18_root, n_fft=n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, sample_rate=sample_rate, sources=sources, target=target)

        valid_txt_path = os.path.join(musdb18_root, 'validation.txt')
        train_txt_path = os.path.join(musdb18_root, 'train.txt')

        with open(valid_txt_path, 'r') as f:
            valid_lst = [line.strip() for line in f]

        names = []

        with open(train_txt_path, 'r') as f:
            for line in f:
                name = line.strip()

                if (not include_valid) and name in valid_lst:
                    continue

                names.append(name)

        self.patch_samples = patch_samples

        self.augmentation = augmentation

        self.tracks = []

        if augmentation:
            duration = patch_samples / sample_rate
            total_duration = 0

            for trackID, name in enumerate(names):
                mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
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
                    track['path'][source] = os.path.join(musdb18_root, 'train', name, "{}.wav".format(source))

                self.tracks.append(track)

                track_duration = track_samples / track_sample_rate
                total_duration += track_duration

            if samples_per_epoch is None:
                samples_per_epoch = int(total_duration / duration)

            self.samples_per_epoch = samples_per_epoch
            self.json_data = None
        else:
            if overlap is None:
                overlap = patch_samples // 2

            self.samples_per_epoch = None

            for trackID, name in enumerate(names):
                mixture_path = os.path.join(musdb18_root, 'train', name, "mixture.wav")
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
            mixture <torch.Tensor>: Complex tensor with shape (1, n_mics, n_bins, n_frames)  if `target` is list, otherwise (n_mics, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), n_mics, n_bins, n_frames) if `target` is list, otherwise (n_mics, n_bins, n_frames)
        """
        if self.augmentation:
            mixture, target = self._getitem_augmentation()
        else:
            mixture, target = self._getitem(idx)

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)
        target = stft(target, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources), n_mics, n_bins, n_frames) or (n_mics, n_bins, n_frames)

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
        mixture, target, _, _ = super().__getitem__(idx)

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
            track_samples = track['samples']

            start = random.randint(0, track_samples - self.patch_samples - 1)
            source, _ = torchaudio.load(source_path, frame_offset=start, num_frames=self.patch_samples)

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

"""
    Data loader
"""
class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

class TestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)

        self.collate_fn = test_collate_fn

def test_collate_fn(batch):
    batched_mixture, batched_sources = None, None
    batched_segment_ID = []

    for mixture, sources, segmend_ID in batch:
        mixture = mixture.unsqueeze(dim=0)
        sources = sources.unsqueeze(dim=0)

        if batched_mixture is None:
            batched_mixture = mixture
            batched_sources = sources
        else:
            batched_mixture = torch.cat([batched_mixture, mixture], dim=0)
            batched_sources = torch.cat([batched_sources, sources], dim=0)

        batched_segment_ID.append(segmend_ID)

    return batched_mixture, batched_sources, batched_segment_ID

def assert_sample_rate(sample_rate):
    assert sample_rate == SAMPLE_RATE_MUSDB18, "sample rate is expected {}, but given {}".format(SAMPLE_RATE_MUSDB18, sample_rate)

def _test_train_dataset():
    torch.manual_seed(111)

    musdb18_root = "../../../../../db/musdb18"

    dataset = WaveTrainDataset(musdb18_root, duration=4, sources=__sources__)
    loader = TrainDataLoader(dataset, batch_size=6, shuffle=True)

    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break

    dataset = SpectrogramTrainDataset(musdb18_root, n_fft=2048, hop_length=512, sample_rate=8000, duration=4, sources=__sources__)
    loader = TrainDataLoader(dataset, batch_size=6, shuffle=True)

    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break

if __name__ == '__main__':
    _test_train_dataset()
