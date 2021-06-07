import os
import json

import numpy as np
import musdb
import torch
from torchaudio.transforms import Spectrogram

__sources__=['drums','bass','other','vocals']

EPS=1e-12
THRESHOLD_POWER=1e-5

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, musdb18_root, sr=44100, target=None):
        super().__init__()

        assert target in __sources__, "target is unknown"
        
        self.musdb18_root = os.path.abspath(musdb18_root)
        self.mus = musdb.DB(root=self.musdb18_root)

        self.sr = sr
        self.target = target

class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sr=44100, target=None):
        super().__init__(musdb18_root, sr=sr, target=target)

        self.json_data = None

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture (2, T) <torch.Tensor>
            target (2, T) <torch.Tensor>
            title <str>: title of song
        """
        data = self.json_data[idx]

        songID = data['songID']
        track = self.mus.tracks[songID]
        title = track.title
        track.sample_rate = self.sr
        track.chunk_start = data['start']
        track.chunk_duration = data['duration']

        mixture = track.audio.transpose(1, 0)
        target = track.targets[self.target].audio.transpose(1, 0)

        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        return mixture, target, title

    def __len__(self):
        return len(self.json_data)
    
    def save_as_json(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)

class SpectrogramDataset(WaveDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, target=None, json_path=None):
        super().__init__(musdb18_root, sr=sr, target=target)
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size
        self.n_bins = fft_size//2 + 1

        if window_fn:
            if window_fn == 'hann':
                window_fn = torch.hann_window
            else:
                raise ValueError("Invalid argument.")
        
        self.stft = Spectrogram(n_fft=fft_size, hop_length=hop_size, window_fn=window_fn, power=None, normalized=normalize)
    
        if json_path is not None:
            with open(json_path, 'r') as f:
                self.json_data = json.load(f)

    def _is_active(self, input, threshold=1e-5):
        input = self.stft(input) # (2, n_bins, n_frames, 2)
        power = torch.sum(input**2, dim=3) # (2, n_bins, n_frames)
        power = torch.mean(power)

        if power.item() >= threshold:
            return True
        else:
            return False
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (2, n_bins, n_frames, 2) <torch.Tensor>, first n_bins is real, the latter n_bins is iamginary part.
            sources (2, n_bins, n_frames, 2) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            title <str>: title of song
        """
        mixture, sources, title = super().__getitem__(idx)
        
        T = mixture.size(-1)

        mixture = self.stft(mixture) # (2, n_bins, n_frames, 2)
        sources = self.stft(sources) # (2, n_bins, n_frames, 2)
        
        return mixture, sources, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path)
        return dataset

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, duration=4, overlap=None, target=None, json_path=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train')

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
                
                track.sample_rate = self.sr
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
            mixture (1, 2, n_bins, n_frames, 2) <torch.Tensor>, first n_bins is real, the latter n_bins is iamginary part.
            sources (n_sources, 2, n_bins, n_frames, 2) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            title <str>: title of song
        """
        mixture, sources, _, _ = super().__getitem__(idx)
        
        return mixture, sources
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

class SpectrogramEvalDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, max_duration=10, target=None, json_path=None):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, target=target)
        
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid')

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
            mixture (2, n_bins, n_frames, 2) <torch.Tensor>, first n_bins is real, the latter n_bins is iamginary part.
            sources (2, n_bins, n_frames, 2) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            title <str>: title of song
        """
        mixture, sources, T, title = super().__getitem__(idx)
        
        return mixture, sources, T, title
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, fft_size, sr=44100, target=None, **kwargs):
        dataset = cls(musdb18_root, fft_size, sr=sr, target=target, json_path=json_path, **kwargs)
        return dataset

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