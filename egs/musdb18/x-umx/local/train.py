#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from utils.augmentation import SequentialAugmentation, choose_augmentation
from dataset import AugmentationSpectrogramTrainDataset, TrainDataLoader
from adhoc_dataset import SpectrogramEvalDataset, EvalDataLoader
from adhoc_driver import AdhocSchedulerTrainer
from models.xumx import CrossNetOpenUnmix
from criterion.distance import MeanSquaredError
from criterion.sdr import NegWeightedSDR
from adhoc_criterion import MultiDomainLoss

parser = argparse.ArgumentParser(description="Training of CrossNet-Open-Unmix")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sample_rate', '-sr', type=int, default=44100, help='Sampling rate')
parser.add_argument('--duration', type=float, default=6, help='Duration')
parser.add_argument('--valid_duration', type=float, default=30, help='Max duration for validation')
parser.add_argument('--n_fft', type=int, default=4096, help='FFT length')
parser.add_argument('--hop_length', type=int, default=1024, help='Hop length')
parser.add_argument('--max_bin', type=int, default=1487, help='Max frequency bin')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--augmentation_path', type=str, default=None, help='Path to augmentation.yaml')
parser.add_argument('--hidden_channels', type=int, default=512, help='# of hidden channels')
parser.add_argument('--num_layers', type=int, default=3, help='# of layers in LSTM')
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--bridge', type=int, default=1, help='Bridging network.')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--combination', type=int, default=1, help='Combination Loss.')
parser.add_argument('--criterion_time', type=str, default='wsdr', choices=['wsdr'], help='Criterion in time domain')
parser.add_argument('--criterion_frequency', type=str, default='mse', choices=['mse'], help='Criterion in time-frequency domain')
parser.add_argument('--weight_time', type=float, default=1, help='Weight for time domain loss')
parser.add_argument('--weight_frequency', type=float, default=10, help='Weight for frequency domain loss')
parser.add_argument('--min_pair', type=int, default=1, help='Minimum pair for combination loss')
parser.add_argument('--max_pair', type=int, default=None, help='Maximum pair for combination loss. If you set None, max_pair is regarded as len(sources) - 1.')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--scheduler_path', type=str, default=None, help='Path to scheduler.yaml')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size. Default: 16')
parser.add_argument('--samples_per_epoch', type=int, default=64*100, help='Training samples in one epoch')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_norbert', type=int, default=0, help='Use norbert.wiener for multichannel wiener filetering. 0: Not use norbert, 1: Use norbert (you have to install norbert)')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--num_workers', type=int, default=0, help='# of workers given to data loader for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    patch_samples = int(args.duration * args.sample_rate)
    max_samples = int(args.valid_duration * args.sample_rate)
    padding = 2 * (args.n_fft // 2)
    patch_size = (patch_samples + padding - args.n_fft) // args.hop_length + 1

    if args.samples_per_epoch <= 0:
        args.samples_per_epoch = None
    
    with open(args.augmentation_path) as f:
        config_augmentation = yaml.safe_load(f)
    
    augmentation = SequentialAugmentation()
    for name in config_augmentation['augmentation']:
        augmentation.append(choose_augmentation(name, **config_augmentation[name]))
    
    train_dataset = AugmentationSpectrogramTrainDataset(
        args.musdb18_root,
        n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn,
        sample_rate=args.sample_rate, patch_samples=patch_samples, samples_per_epoch=args.samples_per_epoch,
        sources=args.sources, target=args.sources,
        include_valid=True,
        augmentation=augmentation
    )
    valid_dataset = SpectrogramEvalDataset(args.musdb18_root, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, sample_rate=args.sample_rate, patch_size=patch_size, max_samples=max_samples, sources=args.sources, target=args.sources)
    
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    in_channels = 2
    args.n_bins = args.n_fft // 2 + 1
    model = CrossNetOpenUnmix(
        in_channels, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
        n_bins=args.n_bins, max_bin=args.max_bin,
        dropout=args.dropout,
        causal=args.causal,
        bridge=args.bridge,
        sources=args.sources
    )

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
        
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))

    # Scheduler
    with open(args.scheduler_path) as f:
        config_scheduler = yaml.safe_load(f)
    
    if config_scheduler['scheduler'] == 'ReduceLROnPlateau':
        config_scheduler.pop('scheduler')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config_scheduler)
    else:
        raise NotImplementedError("Not support schduler {}.".format(args.scheduler))
    
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
            n_fft=args.n_fft, hop_length=args.hop_length, window=train_dataset.window, normalize=train_dataset.normalize,
            source_dim=1, min_pair=args.min_pair, max_pair=args.max_pair # for combination loss
        )
    else:
        criterion = MultiDomainLoss(
            criterion_time, criterion_frequency,
            combination=False,
            weight_time=args.weight_time, weight_frequency=args.weight_frequency,
            n_fft=args.n_fft, hop_length=args.hop_length, window=train_dataset.window, normalize=train_dataset.normalize
        )

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    trainer = AdhocSchedulerTrainer(model, loader, criterion, optimizer, scheduler, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
