import json

import numpy as np
import musdb
import torch

from dataset import MUSDB18Dataset

__sources__ = ['drums', 'bass', 'other', 'vocals']

EPS = 1e-12
THRESHOLD_POWER = 1e-5

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
        track.chunk_start = data['start']
        track.chunk_duration = data['duration']

        mixture = track.audio.transpose(1, 0)[None]
        target = track.targets[self.target].audio.transpose(1, 0)[None]
        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        return mixture, target, title

    def __len__(self):
        return len(self.json_data)
    
    def save_as_json(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.json_data, f, indent=4)

    def _is_active(self, input, threshold=1e-5):
        power = torch.mean(input**2) # (2, T)

        if power.item() >= threshold:
            return True
        else:
            return False

class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, duration=4, overlap=None, target=None, json_path=None, threshold=THRESHOLD_POWER):
        super().__init__(musdb18_root, sr=sr, target=target)

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
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path)
        return dataset

class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, max_duration=4, target=None, json_path=None):
        super().__init__(musdb18_root, sr=sr, target=target)

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
    
    @classmethod
    def from_json(cls, musdb18_root, json_path, sr=44100, target=None):
        dataset = cls(musdb18_root, sr=sr, target=target, json_path=json_path)
        return dataset

def _test_train_dataset():
    torch.manual_seed(111)
    
    musdb18_root = "../../../../../db/musdb18"

    dataset = WaveTrainDataset(musdb18_root, sr=8000, duration=4, target='vocals')
    
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

    dataset.save_as_json('data/tmp.json')

    WaveTrainDataset.from_json(musdb18_root, 'data/tmp.json', sr=44100, target='vocals')
    for mixture, sources in dataset:
        print(mixture.size(), sources.size())
        break

if __name__ == '__main__':
    _test_train_dataset()