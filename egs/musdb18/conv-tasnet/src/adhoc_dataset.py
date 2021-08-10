import json

import numpy as np
import musdb
import torch

from dataset import MUSDB18Dataset

__sources__ = ['drums', 'bass', 'other', 'vocals']


SAMPLE_RATE_MUSDB18 = 44100
EPS = 1e-12
THRESHOLD_POWER = 1e-5

class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, sources=__sources__, target=None):
        """
        Args:
            musdb18_root <int>: Path to MUSDB or MUSDB-HQ
            sr: Sampling frequency. Default: 44100
            sources <list<str>>: Sources included in mixture
            target <str> or <list<str>>: 
        """
        super().__init__(musdb18_root, sr=sr, sources=sources, target=target)

        self.json_data = None

    def __getitem__(self, idx):
        """
        Args:
            idx <int>: index
        Returns:
            mixture <torch.Tensor>: (1, 2, T) if `target` is list, otherwise (2, T)
            target <torch.Tensor>: (len(target), 2, T) if `target` is list, otherwise (2, T)
            name <str>: Artist and title of song
        """
        data = self.json_data[idx]

        songID = data['songID']
        track = self.mus.tracks[songID]
        name = track.name
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

        return mixture, target, name

    def __len__(self):
        return len(self.json_data)
    
    def save_as_json(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)

    def _is_active(self, input, threshold=THRESHOLD_POWER):
        power = torch.mean(input**2) # (2, T)

        if power.item() >= threshold:
            return True
        else:
            return False

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, duration=4, overlap=None, target=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, sr=sr, target=target)

        assert_sample_rate(sr)
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train', is_wav=True)
        
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
        mixture, sources, _ = super().__getitem__(idx)
        
        return mixture, sources

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=SAMPLE_RATE_MUSDB18, max_duration=4, target=None):
        super().__init__(musdb18_root, sr=sr, target=target)

        assert_sample_rate(sr)
        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid', is_wav=True)

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

def _test_train_dataset():
    torch.manual_seed(111)
    
    musdb18_root = "../../../../../db/musdb18"

    dataset = WaveTrainDataset(musdb18_root, sr=8000, duration=4, target='vocals')
    
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

    dataset.save_as_json('data/tmp.json')

    WaveTrainDataset.from_json(musdb18_root, 'data/tmp.json', sr=SAMPLE_RATE_MUSDB18, target='vocals')
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

if __name__ == '__main__':
    _test_train_dataset()