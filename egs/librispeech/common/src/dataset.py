import os
import json
import numpy as np
import soundfile as sf
import torch

from algorithm.stft import BatchSTFT
from algorithm.ideal_mask import ideal_binary_mask, ideal_ratio_mask, wiener_filter_mask

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, json_path):
        super().__init__()
        
        self.wav_root = wav_root
        
        with open(json_path) as f:
            self.json_data = json.load(f)

class WaveDataset(LibriSpeechDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
        """
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

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)


class WaveEvalDataset(WaveDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)


class WaveTestDataset(WaveDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            segment_IDs (n_sources,) <list<str>>
        """
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
        
        return mixture, sources, segment_IDs
    
    def __len__(self):
        return len(self.json_data)


class SpectrogramDataset(WaveDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False):
        super().__init__(wav_root, json_path)
        
        if hop_size is None:
            hop_size = fft_size//2
        
        self.fft_size, self.hop_size = fft_size, hop_size
        self.F_bin = fft_size//2 + 1
        
        self.stft = BatchSTFT(fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, 2*F_bin, T_bin) <torch.Tensor>, first F_bin is real, the latter F_bin is iamginary part.
            sources (n_sources, 2*F_bin, T_bin) <torch.Tensor>
        """
        mixture, sources = super().__getitem__(idx)
        
        mixture = self.stft(mixture) # (1, 2*F_bin, T_bin)
        sources = self.stft(sources) # (n_sources, 2*F_bin, T_bin)
        
        return mixture, sources

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)


class IdealMaskSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)
        
        if mask_type == 'ibm':
            self.generate_mask = ideal_binary_mask
        elif mask_type == 'irm':
            self.generate_mask = ideal_ratio_mask
        elif mask_type == 'wfm':
            self.generate_mask = wiener_filter_mask
        else:
            raise NotImplementedError("Not support mask {}".format(mask_type))
        
        self.threshold = threshold
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, 2*F_bin, T_bin) <torch.Tensor>
            sources (n_sources, 2*F_bin, T_bin) <torch.Tensor>
            ideal_mask (n_sources, F_bin, T_bin) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
        """
        F_bin = self.F_bin
        threshold = self.threshold
        
        mixture, sources = super().__getitem__(idx) # (1, 2*F_bin, T_bin), (n_sources, 2*F_bin, T_bin)
        real, imag = sources[:,:F_bin], sources[:,F_bin:]
        sources_amplitude = torch.sqrt(real**2+imag**2)
        ideal_mask = self.generate_mask(sources_amplitude)
        
        log_amplitude = 20 * torch.log10(sources_amplitude)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(sources_amplitude > 0, torch.ones_like(sources_amplitude), torch.zeros_like(sources_amplitude))
        
        return mixture, sources, ideal_mask, threshold_weight
        

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

