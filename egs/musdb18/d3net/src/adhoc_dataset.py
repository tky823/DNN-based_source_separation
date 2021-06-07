import os
import json

import numpy as np
import musdb
import torch

from dataset import WaveDataset, SpectrogramDataset

__sources__=['drums','bass','other','vocals']

EPS=1e-12
THRESHOLD_POWER=1e-5

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, duration=4, overlap=None, sources=__sources__, target=None, json_path=None, threshold=THRESHOLD_POWER):
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
        
        raise NotImplementedError("TODO augmentation")
        
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

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, sr=44100, duration=4, overlap=None, sources=__sources__, target=None, json_path=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, fft_size=fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, sr=sr, sources=sources, target=target)
        
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
            mixture <torch.Tensor>: Complex tensor with shape (1, 2, n_bins, n_frames)  if `target` is list, otherwise (2, n_bins, n_frames) 
            target <torch.Tensor>: Complex tensor with shape (len(target), 2, n_bins, n_frames) if `target` is list, otherwise (2, n_bins, n_frames)
        """
        mixture, target, _, _ = super().__getitem__(idx)
        
        return mixture, target
    
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