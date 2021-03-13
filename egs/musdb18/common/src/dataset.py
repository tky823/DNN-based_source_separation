import os
import numpy as np
import musdb
import torch

from algorithm.stft import BatchSTFT

sources=['drums','bass','other','vocals']
EPS=1e-12

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, musdb18_root, sr=44100, sources=sources):
        super().__init__()
        
        self.musdb18_root = os.path.abspath(musdb18_root)
        self.mus = musdb.DB(root=self.musdb18_root)

        self.sr = sr
        self.sources = sources


class WaveDataset(MUSDB18Dataset):
    def __init__(self, musdb18_root, sr=44100, sources=sources):
        super().__init__(musdb18_root, sr=sr, sources=sources)

        self.json_data = None

    def __getitem__(self, idx):
        data = self.json_data[idx]

        songID = data['songID']
        track = self.mus.tracks[songID]
        title = track.title
        track.sample_rate = self.sr
        track.chunk_start = data['start']
        track.chunk_duration = data['duration']

        mixture = track.audio.transpose(1, 0)[None]
        target = []
        for source in self.sources:
            target.append(track.targets[source].audio.transpose(1, 0))
        target = np.concatenate([
            target
        ], axis=0)

        mixture = torch.Tensor(mixture).float()
        target = torch.Tensor(target).float()

        print(mixture.size(), target.size())
        exit()

        return mixture, target, title

    def __len__(self):
        return len(self.json_data)


class WaveTrainDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, duration=4, overlap=None, sources=sources):
        super().__init__(musdb18_root, sr=sr, sources=sources)

        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='train')

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
        mixture, sources, _ = super().__getitem__(idx)
        
        return mixture, sources


class WaveEvalDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, max_duration=4, sources=sources):
        super().__init__(musdb18_root, sr=sr, sources=sources)

        self.mus = musdb.DB(root=self.musdb18_root, subsets="train", split='valid')

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


class WaveTestDataset(WaveDataset):
    def __init__(self, musdb18_root, sr=44100, max_duration=4, sources=sources):
        super().__init__(musdb18_root, sr=sr, sources=sources)

        self.mus = musdb.DB(root=self.musdb18_root, subsets="test")

        self.json_data = []

        for songID, track in enumerate(self.mus.tracks):
            data = {
                'songID': songID,
                'start': 0,
                'duration': track.duration
            }
            self.json_data.append(data)


class SpectrogramDataset(WaveDataset):
    def __init__(self, musdb18_root, fft_size, hop_size=None, window_fn='hann', normalize=False, duration=4, overlap=None, sources=sources):
        super().__init__(musdb18_root, duration=duration, overlap=overlap, sources=sources)
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size
        self.n_bins = fft_size//2 + 1
        
        self.stft = BatchSTFT(fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames, 2) <torch.Tensor>, first n_bins is real, the latter n_bins is iamginary part.
            sources (n_sources, n_bins, n_frames, 2) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, title = super().__getitem__(idx)
        
        T = mixture.size(-1)

        mixture = self.stft(mixture) # (1, n_bins, n_frames, 2)
        sources = self.stft(sources) # (n_sources, n_bins, n_frames, 2)
        
        return mixture, sources, T, title

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
    
    dataset = WaveTrainDataset(musdb18_root, duration=4, sources=sources)
    loader = TrainDataLoader(dataset, batch_size=4, shuffle=True)
    
    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break


if __name__ == '__main__':
    _test_train_dataset()
