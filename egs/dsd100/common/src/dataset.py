import os
import glob
import numpy as np
import librosa
import torch

from algorithm.stft import BatchSTFT
from algorithm.frequency_mask import ideal_binary_mask, ideal_ratio_mask, wiener_filter_mask

EPS=1e-12

__sources__ = ['vocals', 'bass', 'drums', 'other']

class DSD100Dataset(torch.utils.data.Dataset):
    def __init__(self, dsd100_root):
        super().__init__()
        
        self.dsd100_root = dsd100_root
        self.sources_dir = os.path.join(dsd100_root, 'Sources/Dev')
        self.mixture_dir = os.path.join(dsd100_root, 'Mixture/Dev')
    
    def _search_titles(self):
        search_path = "{}/*".format(self.sources_dir)
        titles = [os.path.basename(path) for path in glob.glob(search_path)]
        self.titles = sorted(titles)

class WaveDataset(DSD100Dataset):
    def __init__(self, dsd100_root, sources, sr):
        super().__init__(dsd100_root)

        self.sources = sources
        self.sr = sr
    
    def __getitem__(self, idx):
        sr = self.sr
        data = self.json_data[idx]

        title = data['title']
        start_idx, end_idx = data['start'], data['end']
        mixture_data = data['mixture']
        sources_data = data['sources']

        mixture, sr = librosa.load(mixture_data['path'], sr, mono=True)
        mixture = mixture[start_idx: end_idx]
        mixture = mixture[np.newaxis,:]

        sources = []
        for _source in self.sources:
            source, sr = librosa.load(sources_data[_source]['path'], sr, mono=True)
            source = source[start_idx: end_idx]
            sources.append(source)
        sources = np.array(sources)

        mixture = torch.Tensor(mixture).float()
        sources = torch.Tensor(sources).float()

        return mixture, sources, title, start_idx, end_idx
    
    def __len__(self):
        return len(self.json_data)
    
    def _split(self, samples, overlap=None):
        sr = self.sr
        
        if overlap is None:
            overlap = samples // 2

        self.json_data = []

        for title in self.titles:
            wave_path = os.path.join(self.sources_dir, title, 'vocals.wav')
            wave, sr = librosa.load(wave_path, sr, mono=True)

            T = len(wave)

            for start_idx in range(0, T, samples - overlap):
                end_idx = start_idx + samples
                if end_idx > T:
                    break
                data = {
                    'title': title,
                    'start': start_idx,
                    'end': end_idx,
                    'sources': {},
                    'mixture': {}
                }
                    
                for source in __sources__:
                    source_data = {
                        'path': os.path.join(self.sources_dir, title, '{}.wav'.format(source)),
                        
                    }
                    data['sources'][source] = source_data
                
                mixture_data = {
                    'path': os.path.join(self.mixture_dir, title, 'mixture.wav'),
                }

                data['mixture'] = mixture_data
            
                self.json_data.append(data)

# TODO: Fix validation dataset

class WaveTrainDataset(WaveDataset):
    def __init__(self, dsd100_root, sources, sr, samples, overlap=None, n_train=40):
        super().__init__(dsd100_root, sources, sr)

        self.n_train = n_train
        
        self.sources_dir = os.path.join(dsd100_root, 'Sources/Dev')
        self.mixture_dir = os.path.join(dsd100_root, 'Mixtures/Dev')

        self._search_titles()
        self._split(samples, overlap=overlap)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
        """
        mixture, sources, _, _, _ = super().__getitem__(idx)

        return mixture, sources
    
    def _search_titles(self):
        super()._search_titles()

        self.titles = self.titles[:self.n_train]


class WaveEvalDataset(WaveDataset):
    def __init__(self, dsd100_root, sources, sr, samples, overlap=None, n_train=40):
        super().__init__(dsd100_root, sources, sr)

        self.n_train = n_train

        self.sources_dir = os.path.join(dsd100_root, 'Sources/Test')
        self.mixture_dir = os.path.join(dsd100_root, 'Mixtures/Test')

        self._search_titles()
        self._split(samples, overlap=overlap)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            title <str>
        """
        mixture, sources, title, _, _ = super().__getitem__(idx)
    
        return mixture, sources, title
    
    def _search_titles(self):
        super()._search_titles()

        self.titles = self.titles[self.n_train:]


class WaveTestDataset(WaveDataset):
    def __init__(self, dsd100_root, sources, sr, samples, overlap=None):
        super().__init__(dsd100_root, sources, sr)
        
        self.sources_dir = os.path.join(dsd100_root, 'Sources/Test')
        self.mixture_dir = os.path.join(dsd100_root, 'Mixtures/Test')

        self._search_titles()
        self._split(samples, overlap=overlap)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, T) <torch.Tensor>
            sources (n_sources, T) <torch.Tensor>
            title <str>
            start_idx <int>
            end_idx <int>
        """
        mixture, sources, title, start_idx, end_idx = super().__getitem__(idx)

        return mixture, sources, title, start_idx, end_idx

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
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>, first F_bin is real, the latter F_bin is iamginary part.
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, segment_IDs = super().__getitem__(idx)
        
        T = mixture.size(-1)
        
        mixture = self.stft(mixture) # (1, F_bin, T_bin, 2)
        sources = self.stft(sources) # (n_sources, F_bin, T_bin, 2)
        
        return mixture, sources, T, segment_IDs

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)
    
    def __getitem__(self, idx):
        mixture, sources, _, _ = super().__getitem__(idx)
        
        return mixture, sources


class IdealMaskSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, eps=EPS):
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
        self.eps = eps
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            ideal_mask (n_sources, F_bin, T_bin) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        F_bin = self.F_bin
        threshold = self.threshold
        eps = self.eps
        
        mixture, sources, T, segment_IDs = super().__getitem__(idx) # (1, F_bin, T_bin, 2), (n_sources, F_bin, T_bin, 2)
        real, imag = sources[...,0], sources[...,1]
        sources_amplitude = torch.sqrt(real**2+imag**2)
        ideal_mask = self.generate_mask(sources_amplitude)
        
        real, imag = mixture[...,0], mixture[...,1]
        mixture_amplitude = torch.sqrt(real**2+imag**2)
        log_amplitude = 20 * torch.log10(mixture_amplitude + eps)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(mixture_amplitude > 0, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))
        
        return mixture, sources, ideal_mask, threshold_weight, T, segment_IDs


class IdealMaskSpectrogramTrainDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            ideal_mask (n_sources, F_bin, T_bin) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
        """
        mixture, sources, ideal_mask, threshold_weight, _, _ = super().__getitem__(idx)
        
        return mixture, sources, ideal_mask, threshold_weight


class IdealMaskSpectrogramEvalDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold)
    
    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            ideal_mask (n_sources, F_bin, T_bin) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
        """
        mixture, sources, ideal_mask, threshold_weight, _, _ = super().__getitem__(idx)
    
        return mixture, sources, ideal_mask, threshold_weight


class IdealMaskSpectrogramTestDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold)

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            ideal_mask (n_sources, F_bin, T_bin) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
            T () <int>
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, ideal_mask, threshold_weight, T, segment_IDs = super().__getitem__(idx)

        return mixture, sources, ideal_mask, threshold_weight, T, segment_IDs

class ThresholdWeightSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, threshold=40, eps=EPS):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize)
        
        self.threshold = threshold
        self.eps = eps

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        F_bin = self.F_bin
        threshold = self.threshold
        eps = self.eps
        
        mixture, sources, T, segment_IDs = super().__getitem__(idx) # (1, F_bin, T_bin, 2), (n_sources, F_bin, T_bin, 2)
        
        real, imag = mixture[...,0], mixture[...,1]
        mixture_amplitude = torch.sqrt(real**2+imag**2)
        log_amplitude = 20 * torch.log10(mixture_amplitude + eps)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(mixture_amplitude > 0, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))
        
        return mixture, sources, threshold_weight, T, segment_IDs


class ThresholdWeightSpectrogramTrainDataset(ThresholdWeightSpectrogramDataset):
    def __init__(self, wav_root, json_path, fft_size, hop_size=None, window_fn='hann', normalize=False, threshold=40, eps=EPS):
        super().__init__(wav_root, json_path, fft_size, hop_size=hop_size, window_fn=window_fn, normalize=normalize, threshold=threshold, eps=eps)

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, F_bin, T_bin, 2) <torch.Tensor>
            sources (n_sources, F_bin, T_bin, 2) <torch.Tensor>
            threshold_weight (1, F_bin, T_bin) <torch.Tensor>
        """
        mixture, sources, threshold_weight, _, _ = super().__getitem__(idx)
        
        return mixture, sources, threshold_weight


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


class AttractorTestDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert self.batch_size == 1, "batch_size is expected 1, but given {}".format(self.batch_size)
        
        self.collate_fn = attractor_test_collate_fn


def attractor_test_collate_fn(batch):
    batched_mixture, batched_sources, batched_assignment, batched_weight_threshold = None, None, None, None
    batched_T = []
    batched_segment_ID = []
    
    for mixture, sources, assignment, weight_threshold, T, segmend_ID in batch:
        mixture = mixture.unsqueeze(dim=0)
        sources = sources.unsqueeze(dim=0)
        assignment = assignment.unsqueeze(dim=0)
        weight_threshold = weight_threshold.unsqueeze(dim=0)
        
        if batched_mixture is None:
            batched_mixture = mixture
            batched_sources = sources
            batched_assignment = assignment
            batched_weight_threshold = weight_threshold
        else:
            batched_mixture = torch.cat([batched_mixture, mixture], dim=0)
            batched_sources = torch.cat([batched_sources, sources], dim=0)
            batched_assignment = torch.cat([batched_assignment, assignment], dim=0)
            batched_weight_threshold = torch.cat([batched_weight_threshold, weight_threshold], dim=0)
        
        batched_T.append(T)
        batched_segment_ID.append(segmend_ID)
    
    return batched_mixture, batched_sources, batched_assignment, batched_weight_threshold, batched_T, batched_segment_ID
