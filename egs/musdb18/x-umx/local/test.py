#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from adhoc_dataset import SpectrogramTestDataset, TestDataLoader
from adhoc_driver import AdhocTester
from models.xumx import CrossNetOpenUnmix
from criterion.distance import MeanSquaredError
from criterion.sdr import NegWeightedSDR
from adhoc_criterion import MultiDomainLoss

parser = argparse.ArgumentParser(description="Evaluation of CrossNet-Open-Unmix")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sample_rate', '-sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--duration', type=float, default=6, help='Duration')
parser.add_argument('--n_fft', type=int, default=4096, help='FFT length')
parser.add_argument('--hop_length', type=int, default=1024, help='Hop length')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--combination', type=int, default=1, help='Combination Loss.')
parser.add_argument('--criterion_time', type=str, default='wsdr', choices=['wsdr'], help='Criterion in time domain')
parser.add_argument('--criterion_frequency', type=str, default='mse', choices=['mse'], help='Criterion in time-frequency domain')
parser.add_argument('--weight_time', type=float, default=1, help='Weight for time domain loss')
parser.add_argument('--weight_frequency', type=float, default=10, help='Weight for frequency domain loss')
parser.add_argument('--min_pair', type=int, default=1, help='Minimum pair for combination loss')
parser.add_argument('--max_pair', type=int, default=None, help='Maximum pair for combination loss. If you set None, max_pair is regarded as len(sources) - 1.')
parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model.')
parser.add_argument('--estimates_dir', type=str, default=None, help='Estimated sources output directory')
parser.add_argument('--json_dir', type=str, default=None, help='Json directory')
parser.add_argument('--estimate_all', type=int, default=1, help='Estimates all songs. GPU is required if use_cuda=1.')
parser.add_argument('--evaluate_all', type=int, default=1, help='Evaluates all estimations. GPU is NOT required.')
parser.add_argument('--use_norbert', type=int, default=0, help='Use norbert.wiener for multichannel wiener filetering. 0: Not use norbert, 1: Use norbert (you have to install norbert)')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    samples = int(args.duration * args.sample_rate)
    padding = 2 * (args.n_fft // 2)
    patch_size = (samples + padding - args.n_fft) // args.hop_length + 1
    
    test_dataset = SpectrogramTestDataset(args.musdb18_root, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, sample_rate=args.sample_rate, patch_size=patch_size, sources=args.sources, target=args.sources)
    print("Test dataset includes {} samples.".format(len(test_dataset)))
    
    loader = TestDataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = CrossNetOpenUnmix.build_model(args.model_path)
    
    print(model)
    print("# Parameters: {}".format(model.num_parameters), flush=True)
    
    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA")
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA")
    
    # Criterion
    if args.criterion_time == 'wsdr':
        if args.combination:
            criterion_time = NegWeightedSDR(source_dim=1, reduction='mean') # (batch_size, n_sources, in_channels, T)
        else:
            criterion_time = NegWeightedSDR(source_dim=1, reduction='mean', reduction_dim=2) # (batch_size, n_sources, in_channels, T)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion_time))
    
    if args.criterion_frequency == 'mse':
        if args.combination:
            criterion_frequency = MeanSquaredError(dim=(1,2,3)) # (batch_size, in_channels, n_bins, n_frames) for combination loss, be careful.
        else:
            criterion_frequency = MeanSquaredError(dim=(2,3,4)) # (batch_size, n_sources, in_channels, n_bins, n_frames)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion_time))
    
    if args.combination:
        criterion = MultiDomainLoss(
            criterion_time, criterion_frequency,
            combination=True,
            weight_time=args.weight_time, weight_frequency=args.weight_frequency,
            n_fft=args.n_fft, hop_length=args.hop_length, window=test_dataset.window, normalize=test_dataset.normalize,
            source_dim=1, min_pair=args.min_pair, max_pair=args.max_pair # for combination loss
        )
    else:
        criterion = MultiDomainLoss(
            criterion_time, criterion_frequency,
            combination=False,
            weight_time=args.weight_time, weight_frequency=args.weight_frequency,
            n_fft=args.n_fft, hop_length=args.hop_length, window=test_dataset.window, normalize=test_dataset.normalize
        )
    
    tester = AdhocTester(model, loader, criterion, args)
    tester.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
