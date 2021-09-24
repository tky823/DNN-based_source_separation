#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from utils.utils_augmentation import SequentialAugmentation, choose_augmentation
from dataset import TrainDataLoader
from adhoc_dataset import SpectrogramTrainDataset, SpectrogramEvalDataset, EvalDataLoader
from adhoc_driver import AdhocTrainer
from models.umx import OpenUnmix
from criterion.distance import MeanSquaredError

parser = argparse.ArgumentParser(description="Training of Open-Unmix")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--duration', type=float, default=6, help='Duration')
parser.add_argument('--valid_duration', type=float, default=30, help='Max duration for validation')
parser.add_argument('--fft_size', type=int, default=4096, help='FFT length')
parser.add_argument('--hop_size', type=int, default=1024, help='Hop length')
parser.add_argument('--max_bin', type=int, default=1487, help='Max frequency bin')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--augmentation_path', type=str, default=None, help='Path to augmentation.yaml')
parser.add_argument('--hidden_channels', type=int, default=512, help='# of hidden channels')
parser.add_argument('--num_layers', type=int, default=3, help='# of layers in LSTM')
parser.add_argument('--dropout', type=float, default=0, help='dropout')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--target', type=str, default=None, choices=['bass', 'drums', 'other', 'vocals'], help='Target source name')
parser.add_argument('--criterion', type=str, default='mse', choices=['mse'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size. Default: 128')
parser.add_argument('--samples_per_epoch', type=int, default=64*100, help='Training samples in one epoch')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--num_workers', type=int, default=0, help='# of workers given to data loader for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    patch_samples = int(args.duration * args.sr)
    max_samples = int(args.valid_duration * args.sr)
    padding = 2 * (args.fft_size // 2)
    patch_size = (patch_samples + padding - args.fft_size) // args.hop_size + 1

    if args.samples_per_epoch <= 0:
        args.samples_per_epoch = None
    
    with open(args.augmentation_path) as f:
        config_augmentation = yaml.safe_load(f)
    
    augmentation = SequentialAugmentation()
    for name in config_augmentation['augmentation']:
        augmentation.append(choose_augmentation(name, **config_augmentation[name]))
    
    train_dataset = SpectrogramTrainDataset(args.musdb18_root, fft_size=args.fft_size, hop_size=args.hop_size, window_fn=args.window_fn, sr=args.sr, patch_samples=patch_samples, samples_per_epoch=args.samples_per_epoch, sources=args.sources, target=args.target, augmentation=augmentation)
    valid_dataset = SpectrogramEvalDataset(args.musdb18_root, fft_size=args.fft_size, hop_size=args.hop_size, window_fn=args.window_fn, sr=args.sr, patch_size=patch_size, max_samples=max_samples, sources=args.sources, target=args.target)
    
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    in_channels = 2
    args.n_bins = args.fft_size // 2 + 1
    model = OpenUnmix(in_channels, hidden_channels=args.hidden_channels, num_layers=args.num_layers, n_bins=args.n_bins, max_bin=args.max_bin, dropout=args.dropout, causal=args.causal)

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
    
    # Criterion
    if args.criterion == 'mse':
        criterion = MeanSquaredError(dim=(1,2,3))
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    trainer = AdhocTrainer(model, loader, criterion, optimizer, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
