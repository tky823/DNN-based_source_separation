import torch

from dataset import WaveDataset
from adhoc_utils import create_spk_to_idx

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, n_sources=n_sources)

        self.spk_to_idx = create_spk_to_idx(list_path)
    
    def __getitem__(self, idx):
        mixture, sources, segment_ID = super().__getitem__(idx)
        spk = segment_ID.split('_')[0:-1:2]
        spk_idx = []

        for _spk in spk:
            _spk = self.spk_to_idx(_spk)
            spk_idx.append(_spk)
        
        spk_idx = torch.stack(spk_idx, dim=0)
        
        return mixture, sources, spk_idx
