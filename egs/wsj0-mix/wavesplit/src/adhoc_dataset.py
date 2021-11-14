import os

import torch
import torchaudio

from dataset import WaveDataset

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, n_sources=2, spk_to_idx=None):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, n_sources=n_sources)

        self.spk_to_idx = spk_to_idx
    
    def __getitem__(self, idx):
        mixture, sources, segment_ID = super().__getitem__(idx)
        spk = segment_ID.split('_')[0:-1:2]
        spk_idx = []

        for _spk in spk:
            _spk = self.spk_to_idx(_spk)
            spk_idx.append(_spk)
        
        spk_idx = torch.stack(spk_idx, dim=0)
        
        return mixture, sources, spk_idx

class WaveEvalDataset(WaveDataset):
    def __init__(self, wav_root, list_path, max_samples=None, n_sources=2, spk_to_idx=None):
        super().__init__(wav_root, list_path, n_sources=n_sources)

        self.spk_to_idx = spk_to_idx

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)

        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))

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

        spk = segment_ID.split('_')[0:-1:2]
        spk_idx = []

        for _spk in spk:
            _spk = self.spk_to_idx(_spk)
            spk_idx.append(_spk)
        
        spk_idx = torch.stack(spk_idx, dim=0)
    
        return mixture, sources, spk_idx, segment_ID

