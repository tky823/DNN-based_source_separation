import os
import json

import torch
import torchaudio

from utils.audio import build_window
from algorithm.frequency_mask import compute_ideal_binary_mask, compute_ideal_ratio_mask, compute_wiener_filter_mask

EPS = 1e-12

class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, wav_root, json_path):
        super().__init__()

        self.wav_root = os.path.abspath(wav_root)
        json_path = os.path.abspath(json_path)

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
            segment_IDs (n_sources,) <list<str>>
        """
        data = self.json_data[idx]['sources']
        mixture = 0
        sources = None
        segment_IDs = []

        for key in data.keys():
            source_data = data[key]
            start, end = source_data['start'], source_data['end']
            wav_path = os.path.join(self.wav_root, source_data['path'])
            wave, _ = torchaudio.load(wav_path, frame_offset=start, num_frames=end - start)
            mixture = mixture + wave

            if sources is None:
                sources = wave
            else:
                sources = torch.cat([sources, wave], dim=0)

            segment_IDs.append("{}_{}-{}".format(source_data['utterance-ID'], start, end))

        return mixture, sources, segment_IDs

    def __len__(self):
        return len(self.json_data)

class WaveTrainDataset(WaveDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)

    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        return mixture, sources

class WaveEvalDataset(WaveDataset):
    def __init__(self, wav_root, json_path):
        super().__init__(wav_root, json_path)

    def __getitem__(self, idx):
        mixture, sources, _ = super().__getitem__(idx)
        return mixture, sources

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
        mixture, sources, segment_IDs = super().__getitem__(idx)
        return mixture, sources, segment_IDs

class SpectrogramDataset(WaveDataset):
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False):
        super().__init__(wav_root, json_path)

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
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, segment_IDs = super().__getitem__(idx)

        T = mixture.size(-1)

        mixture = torch.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (1, n_bins, n_frames)
        sources = torch.stft(sources, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window, normalized=self.normalize, return_complex=True) # (n_sources, n_bins, n_frames)

        return mixture, sources, T, segment_IDs

class SpectrogramTrainDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize)

    def __getitem__(self, idx):
        mixture, sources, _, _ = super().__getitem__(idx)
        return mixture, sources

class IdealMaskSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40, eps=EPS):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize)

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
            segment_IDs (n_sources,) <list<str>>
        """
        threshold = self.threshold
        eps = self.eps

        mixture, sources, T, segment_IDs = super().__getitem__(idx) # (1, n_bins, n_frames), (n_sources, n_bins, n_frames)
        sources_amplitude = torch.abs(sources)
        ideal_mask = self.generate_mask(sources_amplitude, source_dim=0)

        mixture_amplitude = torch.abs(mixture)
        log_amplitude = 20 * torch.log10(mixture_amplitude + eps)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(mixture_amplitude > threshold, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))

        return mixture, sources, ideal_mask, threshold_weight, T, segment_IDs

class IdealMaskSpectrogramTrainDataset(IdealMaskSpectrogramDataset):
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold)

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
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold)

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
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False, mask_type='ibm', threshold=40):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, mask_type=mask_type, threshold=threshold)

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            ideal_mask (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
            T () <int>
            segment_IDs (n_sources,) <list<str>>
        """
        mixture, sources, ideal_mask, threshold_weight, T, segment_IDs = super().__getitem__(idx)

        return mixture, sources, ideal_mask, threshold_weight, T, segment_IDs

class ThresholdWeightSpectrogramDataset(SpectrogramDataset):
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False, threshold=40, eps=EPS):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize)

        self.threshold = threshold
        self.eps = eps

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
            T (), <int>: Number of samples in time-domain
            segment_IDs (n_sources,) <list<str>>
        """
        threshold = self.threshold
        eps = self.eps

        mixture, sources, T, segment_IDs = super().__getitem__(idx) # (1, n_bins, n_frames), (n_sources, n_bins, n_frames)

        mixture_amplitude = torch.abs(mixture)
        log_amplitude = 20 * torch.log10(mixture_amplitude + eps)
        max_log_amplitude = torch.max(log_amplitude)
        threshold = 10**((max_log_amplitude - threshold) / 20)
        threshold_weight = torch.where(mixture_amplitude > threshold, torch.ones_like(mixture_amplitude), torch.zeros_like(mixture_amplitude))

        return mixture, sources, threshold_weight, T, segment_IDs

class ThresholdWeightSpectrogramTrainDataset(ThresholdWeightSpectrogramDataset):
    def __init__(self, wav_root, json_path, n_fft, hop_length=None, window_fn='hann', normalize=False, threshold=40, eps=EPS):
        super().__init__(wav_root, json_path, n_fft, hop_length=hop_length, window_fn=window_fn, normalize=normalize, threshold=threshold, eps=eps)

    def __getitem__(self, idx):
        """
        Returns:
            mixture (1, n_bins, n_frames) <torch.Tensor>
            sources (n_sources, n_bins, n_frames) <torch.Tensor>
            threshold_weight (1, n_bins, n_frames) <torch.Tensor>
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
