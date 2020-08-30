import os
import json
import numpy as np
import soundfile as sf
import torch

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, json_path):
        super().__init__()
        
        self.wav_root = wav_root
        
        with open(json_path) as f:
            self.json_data = json.load(f)
        
    def __getitem__(self, idx):
        data = self.json_data[idx]
        mixture = 0
        sources = None
        
        for key in data.keys():
            source_data = data[key]
            start, end = source_data['start'], source_data['end']
            wav_path = os.path.join(self.wav_root, source_data['path'])
            wave, sr = sf.read(wav_path)
            wave = np.array(wave)[start: end]
            wave = wave[None]
            mixture = mixture + wave
        
            if sources is None:
                sources = wave
            else:
                sources = np.concatenate([sources, wave], axis=0)
        
        mixture = torch.Tensor(mixture).float()
        sources = torch.Tensor(sources).float()
        
        return mixture, sources
        
    def __len__(self):
        return len(self.json_data)


class EvalDataset(TrainDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, json_path):
        super().__init__()
        
        self.wav_root = wav_root
        
        with open(json_path) as f:
            self.json_data = json.load(f)
        
    def __getitem__(self, idx):
        data = self.json_data[idx]
        mixture = 0
        sources = None
        segment_IDs = []
        
        for key in data.keys():
            source_data = data[key]
            start, end = source_data['start'], source_data['end']
            wav_path = os.path.join(self.wav_root, source_data['path'])
            wave, sr = sf.read(wav_path)
            wave = np.array(wave)[start: end]
            wave = wave[None]
            mixture = mixture + wave
        
            if sources is None:
                sources = wave
            else:
                sources = np.concatenate([sources, wave], axis=0)
            
            segment_IDs.append("{}_{}-{}".format(source_data['utterance-ID'], start, end))
        
        mixture = torch.Tensor(mixture).float()
        sources = torch.Tensor(sources).float()
        
        print("dataset", segment_IDs)
        
        return mixture, sources, segment_IDs
    
    def __len__(self):
        return len(self.json_data)


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

