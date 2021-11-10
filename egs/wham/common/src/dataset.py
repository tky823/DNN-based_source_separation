import os

import torch
import torchaudio

EPS = 1e-12

class WSJ0Dataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, list_path, task='separate-noisy'):
        super().__init__()
        
        self.wav_root = os.path.abspath(wav_root)
        self.list_path = os.path.abspath(list_path)
        
        if not task in ['enhance', 'separate-noisy']:
            raise ValueError("`task` is expected 'enhance' or 'separate-noisy', but given {}.".format(task))
        
        self.task = task

class WaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, task='separate-noisy', samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, task=task)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)
        
        if overlap is None:
            overlap = samples // 2
        
        if task == 'enhance':
            if n_sources == 1:
                mix_type = 'single'
            elif n_sources == 2:
                mix_type = 'both'
            else:
                raise ValueError("n_sources is expected 1 or 2 in enhancement task, but given {}.".format(n_sources))
        elif task == 'separate-noisy':
            if n_sources == 2:
                mix_type = 'both'
            else:
                raise ValueError("n_sources is expected 2 in separation task, but given {}.".format(n_sources))
        else:
            raise ValueError("`task` is expected 'enhance' or 'separate-noisy', but given {}.".format(task))
        
        self.mix_type = mix_type
        
        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix_{}'.format(mix_type), '{}.wav'.format(ID))
                
                wave, _ = torchaudio.load(wav_path)
                _, T_total = wave.size()
                
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
                            'path': os.path.join('s{}'.format(source_idx + 1), '{}.wav'.format(ID)),
                            'start': start_idx,
                            'end': end_idx
                        }
                        data['sources']['s{}'.format(source_idx + 1)] = source_data
                    
                    noise_data = {
                        'path': os.path.join('noise', '{}.wav'.format(ID)),
                        'start': start_idx,
                        'end': end_idx
                    }

                    data['noise'] = noise_data

                    mixture_data = {
                        'path': os.path.join('mix_{}'.format(mix_type), '{}.wav'.format(ID)),
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
            wave, _ = torchaudio.load(wav_path, frame_offset=start, num_frames=end-start)
            sources.append(wave)
        
        sources = torch.cat(sources, dim=0)

        noise_data = data['mixture']
        start, end = noise_data['start'], noise_data['end']
        wav_path = os.path.join(self.wav_root, noise_data['path'])
        noise, _ = torchaudio.load(wav_path, frame_offset=start, num_frames=end-start)

        mixture_data = data['mixture']
        start, end = mixture_data['start'], mixture_data['end']
        wav_path = os.path.join(self.wav_root, mixture_data['path'])
        mixture, _ = torchaudio.load(wav_path, frame_offset=start, num_frames=end-start)
        
        segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)
        
        return mixture, sources, noise, segment_ID
        
    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, list_path, task='separate-noisy', samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, task=task, samples=samples, overlap=overlap, n_sources=n_sources)
    
    def __getitem__(self, idx):
        mixture, sources, _, _ = super().__getitem__(idx)
        
        return mixture, sources

class WaveEvalDataset(WaveDataset):
    def __init__(self, wav_root, list_path, task='separate-noisy', max_samples=None, n_sources=2):
        super().__init__(wav_root, list_path, task=task, n_sources=n_sources)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)

        mix_type = self.mix_type

        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix_{}'.format(mix_type), '{}.wav'.format(ID))
                wave, _ = torchaudio.load(wav_path)
                _, T_total = wave.size()
                
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
                        'path': os.path.join('s{}'.format(source_idx + 1), '{}.wav'.format(ID)),
                        'start': 0,
                        'end': samples
                    }
                    data['sources']['s{}'.format(source_idx + 1)] = source_data
                
                noise_data = {
                    'path': os.path.join('noise', '{}.wav'.format(ID)),
                    'start': 0,
                    'end': samples
                }

                data['noise'] = noise_data
                
                mixture_data = {
                    'path': os.path.join('mix_{}'.format(mix_type), '{}.wav'.format(ID)),
                    'start': 0,
                    'end': samples
                }
                data['mixture'] = mixture_data
                data['ID'] = ID
            
                self.json_data.append(data)
    
    def __getitem__(self, idx):
        mixture, sources, _, _ = super().__getitem__(idx)
        segment_ID = self.json_data[idx]['ID']

        return mixture, sources, segment_ID

class WaveTestDataset(WaveEvalDataset):
    def __init__(self, wav_root, list_path, task='separate-noisy', max_samples=None, n_sources=2):
        super().__init__(wav_root, list_path, task=task, max_samples=max_samples, n_sources=n_sources)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_ID <str>
        """
        mixture, sources, segment_ID = super().__getitem__(idx)
        
        return mixture, sources, segment_ID

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
