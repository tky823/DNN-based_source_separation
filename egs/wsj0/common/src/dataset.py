import os
import numpy as np
import torch

from utils.utils_audio import read_wav

EPS=1e-12

class WSJ0Dataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, list_path, n_sources=2):
        super().__init__()
        
        self.wav_root = wav_root
        self.list_path = list_path


class WaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, n_sources=n_sources)
        
        if overlap is None:
            overlap = samples//2
        
        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
                y, sr = read_wav(wav_path)
                
                T_total = len(y)
                
                for start_idx in range(0, T_total, samples - overlap):
                    end_idx = start_idx + samples
                    if end_idx > T_total:
                        break
                    data = {
                        'sources': {},
                        'mixture': {}
                    }
                    
                    for source_idx in range(n_sources):
                        source_data = {
                            'path': os.path.join('s{}'.format(source_idx+1), '{}.wav'.format(ID)),
                            'start': start_idx,
                            'end': end_idx
                        }
                        data['sources']['s{}'.format(source_idx+1)] = source_data
                    
                    mixture_data = {
                        'path': os.path.join('mix', '{}.wav'.format(ID)),
                        'start': start_idx,
                        'end': end_idx
                    }
                    data['mixture'] = mixture_data
                    data['ID'] = ID
                
                    self.json_data.append(data)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_IDs (n_sources,) <list<str>>
        """
        data = self.json_data[idx]
        sources = []
        
        for key in data['sources'].keys():
            source_data = data['sources'][key]
            start, end = source_data['start'], source_data['end']
            wav_path = os.path.join(self.wav_root, source_data['path'])
            wave, sr = read_wav(wav_path)
            wave = np.array(wave)[start: end]
            wave = wave[None]
            sources.append(wave)
        
        sources = np.concatenate(sources, axis=0)
        
        mixture_data = data['mixture']
        start, end = mixture_data['start'], mixture_data['end']
        wav_path = os.path.join(self.wav_root, mixture_data['path'])
        wave, sr = read_wav(wav_path)
        wave = np.array(wave)[start: end]
        mixture = wave[None]
            
        segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)
        
        mixture = torch.Tensor(mixture).float()
        sources = torch.Tensor(sources).float()
        
        return mixture, sources, segment_ID
        
    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, n_sources=n_sources)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        
        return mixture, sources


class WaveEvalDataset(WaveDataset):
    def __init__(self, wav_root, list_path, max_samples=None, n_sources=2):
        super().__init__(wav_root, list_path, n_sources=n_sources)
        
        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
                y, sr = read_wav(wav_path)
                
                T_total = len(y)
                
                if max_samples is None:
                    samples = T_total
                else:
                    if T_total < max_samples:
                        samples = T_total
                    else:
                        samples = max_samples
                
                data = {
                    'sources': {},
                    'mixture': {}
                }
                
                for source_idx in range(n_sources):
                    source_data = {
                        'path': os.path.join('s{}'.format(source_idx+1), '{}.wav'.format(ID)),
                        'start': 0,
                        'end': samples
                    }
                    data['sources']['s{}'.format(source_idx+1)] = source_data
                
                mixture_data = {
                    'path': os.path.join('mix', '{}.wav'.format(ID)),
                    'start': 0,
                    'end': samples
                }
                data['mixture'] = mixture_data
                data['ID'] = ID
            
                self.json_data.append(data)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        segment_ID = self.json_data[idx]['ID']
    
        return mixture, sources, segment_ID


class WaveTestDataset(WaveEvalDataset):
    def __init__(self, wav_root, list_path, max_samples=None, n_sources=2):
        super().__init__(wav_root, list_path, max_samples=max_samples, n_sources=n_sources)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_ID <str>
        """
        mixture, sources, segment_ID = super().__getitem__(idx)
        
        return mixture, sources, segment_ID


class TrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EvalDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)


if __name__ == '__main__':
    torch.manual_seed(111)
    
    n_sources = 2
    data_type = 'tt'
    min_max = 'max'
    wav_root = "../../../../../db/wsj0-mix/{}speakers/wav8k/{}/{}".format(n_sources, min_max, data_type)
    list_path = "../../../../dataset/wsj0-mix/{}speakers/mix_{}_spk_{}_{}_mix".format(n_sources, n_sources, min_max, data_type)
    
    dataset = WaveTrainDataset(wav_root, list_path, n_sources=n_sources)
    loader = TrainDataLoader(dataset, batch_size=4, shuffle=True)
    
    for mixture, sources in loader:
        print(mixture.size(), sources.size())
        break
    
    dataset = WaveTestDataset(wav_root, list_path, n_sources=n_sources)
    loader = EvalDataLoader(dataset, batch_size=1, shuffle=False)
    
    for mixture, sources, segment_ID in loader:
        print(mixture.size(), sources.size())
        print(segment_ID)
        break
