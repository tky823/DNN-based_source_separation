import os
import json

import numpy as np
import musdb
import torch

__sources__=['drums','bass','other','vocals']

EPS=1e-12
THRESHOLD_POWER=1e-5

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, musdb18_root, sr=44100, sources=__sources__, target=None):
        """
        Args:
            musdb18_root <str>: Path to MUSDB18 root.
            sr <int>: Sampling rate.
            sources <list<str>>: Sources for mixture. Default: ['drums','bass','other','vocals']
            target <str> or <list<str>>: Target source(s). If None is given, `sources` is used by default.
        """
        super().__init__()

        if target is not None:
            if type(target) is list:
                for _target in target:
                    assert _target in sources, "`sources` doesn't contain target {}".format(_target)
            else:
                assert target in sources, "`sources` doesn't contain target {}".format(target)
        else:
            target = sources
        
        self.musdb18_root = os.path.abspath(musdb18_root)
        self.mus = musdb.DB(root=self.musdb18_root, sample_rate=sr)

        self.sources = sources
        self.target = target

class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sr=44100, sources=__sources__, target=None):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)

        self.json_data = None

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
            title <str>: Title of song
        """
        data = self.json_data[idx]

        songID = data['songID']
        track = self.mus.tracks[songID]
        title = track.title
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

        return mixture, target, title

    def __len__(self):
        return len(self.json_data)
    
    def save_as_json(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, duration=4, overlap=None, sources=__sources__, target=None, json_path=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train', sample_rate=sr)

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return
        
        self.threshold = threshold
        self.duration = duration

        if overlap is None:
            overlap = self.duration / 2

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            for start in np.arange(0, track.duration, duration - overlap):
                if start + duration >= track.duration:
                    break
                data = {
                    'songID': songID,
                    'start': start,
                    'duration': duration
                }
                self.json_data.append(data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
            title <str>: Title of song
        """
        mixture, target, _ = super().__getitem__(idx)
        
        return mixture, target
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, max_duration=10, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', sample_rate=sr)

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return

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
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
            title <str>: Title of song
        """
        mixture, target, title = super().__getitem__(idx)
        
        return mixture, target, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class WaveTestDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, sources=__sources__):
        super().__init__(musdb18_root, sr=sr, sources=sources)

        self.mus = musdb.DB(root=self.musdb18_root, subsets="test", sample_rate=sr)

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            data = {
                'songID': songID,
                'start': 0,
                'duration': track.duration
            }
            self.json_data.append(data)

class SpectrogramDataset(WaveDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size
        self.n_bins = fft_size//2 + 1

        if window_fn:
            if window_fn == 'hann':
                self.window = torch.hann_window(fft_size)
            else:
                raise ValueError("Invalid argument.")
        else:
            self.window = None
        
        self.normalize = normalize
    
        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)

    def _is_active(self, input, threshold=1e-5):
        n_dims = input.dim()

        if n_dims > 2:
            input = input.reshape(-1, input.size(-1))

        input = torch.stft(input, n_fft=self.fft_size, hop_length=self.hop_size, window=self.window, normalized=self.normalize, return_complex=True) # (len(sources)*2, n_bins, n_frames)
        power = torch.sum(torch.abs(input)**2, dim=-1) # (len(sources)*2, n_bins, n_frames)
        power = torch.mean(power)

        if power.item() >= threshold:
            return True
        else:
            return False
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            title <str>: Title of song
        """
        mixture, target, title = super().__getitem__(idx)
        
        n_dims = mixture.dim()
        T = mixture.size(-1)

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

        return mixture, target, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path)
        return dataset

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, duration=4, overlap=None, sources=__sources__, target=None, json_path=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train', sample_rate=sr)

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return
        
        self.threshold = threshold
        self.duration = duration

        if overlap is None:
            overlap = self.duration / 2

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            for start in np.arange(0, track.duration, duration - overlap):
                if start + duration >= track.duration:
                    break
                
                track.chunk_start = start
                track.chunk_duration = duration
                target = track.targets[self.target].audio.transpose(1, 0)
                target = torch.Tensor(target).float()

                if self._is_active(target, threshold=self.threshold):
                    data = {
                        'songID': songID,
                        'start': start,
                        'duration': duration
                    }
                    self.json_data.append(data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
        """
        mixture, target, _, _ = super().__getitem__(idx)
        
        return mixture, target
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, max_duration=10, sources=__sources__, target=None, json_path=None):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', sample_rate=sr)

        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)
            return

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
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
            T (), <int>: Number of samples in time-domain
            title <str>: Title of song
        """
        mixture, sources, T, title = super().__getitem__(idx)
        
        return mixture, sources, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

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

def _test_train_dataset():
    torch.manual_seed(111)
    
    musdb18_root = "../../../../../db/musdb18"
    
    dataset = WaveTrainDataset(musdb18_root, duration=4, sources=__sources__)
    loader = TrainDataLoader(dataset, batch_size=6, shuffle=True)
    
    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break

    dataset = SpectrogramTrainDataset(musdb18_root, fft_size=2048, hop_size=512, sr=8000, duration=4, sources=__sources__)
    loader = TrainDataLoader(dataset, batch_size=6, shuffle=True)
    
    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break


if __name__ == '__main__':
    _test_train_dataset()
