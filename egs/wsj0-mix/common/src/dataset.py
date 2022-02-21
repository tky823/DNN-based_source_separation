import os

import torch
import torchaudio
import torch.nn as nn

from utils.audio import build_window
from transforms.stft import stft
from algorithm.frequency_mask import compute_ideal_binary_mask, compute_ideal_ratio_mask, compute_wiener_filter_mask

EPS = 1e-12

class WSJ0Dataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, list_path):
        super().__init__()
        
        self.wav_root = os.path.abspath(wav_root)
        self.list_path = os.path.abspath(list_path)

class WaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)
        
        if overlap is None:
            overlap = samples // 2
        
        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
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
            segment_ID <str>
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
        
        mixture_data = data['mixture']
        start, end = mixture_data['start'], mixture_data['end']
        wav_path = os.path.join(self.wav_root, mixture_data['path'])
        mixture, _ = torchaudio.load(wav_path, frame_offset=start, num_frames=end-start)
            
        segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)
        
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

class SpectrogramDataset(WaveDataset):
    def __init__(self, wav_root, list_path, n_fft, hop_length=None, window_fn='hann', normalize=False, samples=32000, overlap=None, n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, n_sources=n_sources)
        
        if hop_length is None:
            hop_length = n_fft // 2
        
        self.n_fft, self.hop_length = n_fft, hop_length
        self.n_bins = n_fft // 2 + 1

        if window_fn:
            self.window = build_window(n_fft, window_fn=window_fn)
        else:
            self.window = None
        
        self.normalize = normalize
        
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_ID <str>
        """
        mixture, sources, segment_ID = super().__getitem__(idx)

        T = mixture.size(-1)

        mixture = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_bins, n_frames)
        sources = stft(sources, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (n_sources, n_bins, n_frames)
        
        return mixture, sources, T, segment_ID

class IdealMaskSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, list_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, samples=32000, overlap=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, samples=samples, overlap=overlap, n_sources=n_sources)
        
        if mask_type == 'ibm':
            self.generate_mask = compute_ideal_binary_mask
        elif mask_type == 'irm':
            self.generate_mask = compute_ideal_ratio_mask
        elif mask_type == 'wfm':
            self.generate_mask = compute_wiener_filter_mask
        else:
            raise NotImplementedError("Not support mask {}".format(mask_type))
        
        self.threshold = threshold
        self.eps = eps
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_ID <str>
        """
        threshold = self.threshold
        eps = self.eps
        
        mixture, sources, T, segment_ID = super().__getitem__(idx) # (1, n_bins, n_frames), (n_sources, n_bins, n_frames)
        sources_amplitude = torch.abs(sources)
        ideal_mask = self.generate_mask(sources_amplitude)
        
        mixture_amplitude = torch.abs(mixture)
        log_amplitude = 20 * torch.log10(mixture_amplitude + eps)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(mixture_amplitude > threshold, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))
        
        return mixture, sources, ideal_mask, threshold_weight, T, segment_ID

class IdealMaskSpectrogramTrainDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, list_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, samples=32000, overlap=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold, samples=samples, overlap=overlap, n_sources=n_sources, eps=eps)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
        """
        mixture, sources, ideal_mask, threshold_weight, _, _ = super().__getitem__(idx)
        
        return mixture, sources, ideal_mask, threshold_weight

class IdealMaskSpectrogramEvalDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, list_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, max_samples=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold, n_sources=n_sources, eps=eps)

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
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
        """
        mixture, sources, ideal_mask, threshold_weight, _, _ = super().__getitem__(idx)
    
        return mixture, sources, ideal_mask, threshold_weight

class IdealMaskSpectrogramTestDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, list_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, max_samples=None, n_sources=2, eps=EPS):
        super().__init__(wav_root, list_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold, n_sources=n_sources, eps=eps)

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
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
            T () <int>
            segment_ID (n_sources,) <list<str>>
        """
        mixture, sources, ideal_mask, threshold_weight, T, _ = super().__getitem__(idx)
        segment_ID = self.json_data[idx]['ID']

        return mixture, sources, ideal_mask, threshold_weight, T, segment_ID

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
    batched_mixture, batched_sources = [], []
    batched_segment_ID = []
    
    for mixture, sources, segmend_ID in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)
        batched_segment_ID.append(segmend_ID)
    
    batched_mixture = torch.stack(batched_mixture, dim=0)
    batched_sources = torch.stack(batched_sources, dim=0)
    
    return batched_mixture, batched_sources, batched_segment_ID

class IdealMaskSpectrogramTestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)
        
        self.collate_fn = ideal_mask_spectrogram_test_collate_fn

def ideal_mask_spectrogram_test_collate_fn(batch):
    batched_mixture, batched_sources, batched_ideal_mask, batched_weight_threshold = [], [], [], []
    batched_T = []
    batched_segment_ID = []

    for mixture, sources, ideal_mask, weight_threshold, T, segmend_ID in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)
        batched_ideal_mask.append(ideal_mask)
        batched_weight_threshold.append(weight_threshold)

        batched_T.append(T)
        batched_segment_ID.append(segmend_ID)
    
    batched_mixture = torch.stack(batched_mixture, dim=0)
    batched_sources = torch.stack(batched_sources, dim=0)
    batched_ideal_mask = torch.stack(batched_ideal_mask, dim=0)
    batched_weight_threshold = torch.stack(batched_weight_threshold, dim=0)
    
    return batched_mixture, batched_sources, batched_ideal_mask, batched_weight_threshold, batched_T, batched_segment_ID

class AttractorTestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)
        
        self.collate_fn = attractor_test_collate_fn

def attractor_test_collate_fn(batch):
    batched_mixture, batched_sources, batched_assignment, batched_weight_threshold = [], [], [], []
    batched_T = []
    batched_segment_ID = []
    
    for mixture, sources, assignment, weight_threshold, T, segmend_ID in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)
        batched_assignment.append(assignment)
        batched_weight_threshold.append(weight_threshold)

        batched_T.append(T)
        batched_segment_ID.append(segmend_ID)
    
    batched_mixture = torch.stack(batched_mixture, dim=0)
    batched_sources = torch.stack(batched_sources, dim=0)
    batched_assignment = torch.stack(batched_assignment, dim=0)
    batched_weight_threshold = torch.stack(batched_weight_threshold, dim=0)
    
    return batched_mixture, batched_sources, batched_assignment, batched_weight_threshold, batched_T, batched_segment_ID

"""
Dataset for unknown number of sources.
"""
class MixedNumberSourcesWaveDataset(WSJ0Dataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, max_n_sources=3):
        super().__init__(wav_root, list_path)

        wav_root = os.path.abspath(wav_root)
        list_path = os.path.abspath(list_path)
        
        if overlap is None:
            overlap = samples//2
        
        self.json_data = []
        
        with open(list_path) as f:
            for line in f:
                ID = line.strip()
                wav_path = os.path.join(wav_root, 'mix', '{}.wav'.format(ID))
                
                wave, _ = torchaudio.load(wav_path)
                _, T_total = wave.size()

                n_sources = 0

                for source_idx in range(max_n_sources):
                    wav_path = os.path.join(wav_root, 's{}'.format(source_idx+1), '{}.wav'.format(ID))
                    if not os.path.exists(wav_path):
                        break
                    n_sources += 1
                
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
            segment_ID <str>
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
        
        mixture_data = data['mixture']
        start, end = mixture_data['start'], mixture_data['end']
        wav_path = os.path.join(self.wav_root, mixture_data['path'])
        mixture, _ = torchaudio.load(wav_path, frame_offset=start, num_frames=end-start)
            
        segment_ID = self.json_data[idx]['ID'] + '_{}-{}'.format(start, end)
        
        return mixture, sources, segment_ID
        
    def __len__(self):
        return len(self.json_data)

class MixedNumberSourcesWaveTrainDataset(MixedNumberSourcesWaveDataset):
    def __init__(self, wav_root, list_path, samples=32000, overlap=None, max_n_sources=2):
        super().__init__(wav_root, list_path, samples=samples, overlap=overlap, max_n_sources=max_n_sources)
    
    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        
        return mixture, sources

class MixedNumberSourcesWaveEvalDataset(MixedNumberSourcesWaveDataset):
    def __init__(self, wav_root, list_path, max_samples=None, max_n_sources=3):
        super().__init__(wav_root, list_path, max_n_sources=max_n_sources)

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
                
                n_sources = 0

                for source_idx in range(max_n_sources):
                    wav_path = os.path.join(wav_root, 's{}'.format(source_idx+1), '{}.wav'.format(ID))
                    if not os.path.exists(wav_path):
                        break
                    n_sources += 1
                
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

class MixedNumberSourcesTrainDataLoader(TrainDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.collate_fn = mixed_number_sources_train_collate_fn

class MixedNumberSourcesEvalDataLoader(EvalDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.collate_fn = mixed_number_sources_eval_collate_fn

def mixed_number_sources_train_collate_fn(batch):
    batched_mixture, batched_sources = [], []

    for mixture, sources in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)

    batched_mixture = nn.utils.rnn.pad_sequence(batched_mixture, batch_first=True)
    batched_sources = nn.utils.rnn.pack_sequence(batched_sources, enforce_sorted=False) # n_sources is different from data to data
    
    return batched_mixture, batched_sources

def mixed_number_sources_eval_collate_fn(batch):
    batched_mixture, batched_sources, segment_ID = [], [], []
    batched_segment_ID = []

    for mixture, sources, segment_ID in batch:
        batched_mixture.append(mixture)
        batched_sources.append(sources)
        batched_segment_ID.append(segment_ID)

    batched_mixture = nn.utils.rnn.pad_sequence(batched_mixture, batch_first=True)
    batched_sources = nn.utils.rnn.pack_sequence(batched_sources, enforce_sorted=False) # n_sources is different from data to data
    
    return batched_mixture, batched_sources, batched_segment_ID

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
