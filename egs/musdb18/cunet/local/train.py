#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import TrainDataLoader
from adhoc_dataset import SpectrogramTrainDataset, SpectrogramEvalDataset, EvalDataLoader
from adhoc_driver import AdhocTrainer
from models.cunet import ConditionedUNet2d, ControlDenseNet, UNet2d
from criterion.distance import L1Loss

parser = argparse.ArgumentParser(description="Training of Conditioned UNet")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--config_path', type=str, default=None, help='Path to model configuration file')
parser.add_argument('--sample_rate', '-sr', type=int, default=44100, help='Sampling rate')
parser.add_argument('--patch_size', type=int, default=128, help='Patch size')
parser.add_argument('--valid_duration', type=float, default=10, help='Max duration for validation')
parser.add_argument('--n_fft', type=int, default=4096, help='FFT length')
parser.add_argument('--hop_length', type=int, default=1024, help='Hop length')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--criterion', type=str, default='l1loss', choices=['l1loss'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 4')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    patch_samples = args.hop_length * (args.patch_size - 1) + args.n_fft - 2 * (args.n_fft // 2)
    max_samples = int(args.valid_duration * args.sample_rate)
    
    train_dataset = SpectrogramTrainDataset(args.musdb18_root, n_fft=args.n_fft, hop_length=args.hop_length, sample_rate=args.sample_rate, patch_samples=patch_samples, sources=args.sources, target=args.sources)
    valid_dataset = SpectrogramEvalDataset(args.musdb18_root, n_fft=args.n_fft, hop_length=args.hop_length, sample_rate=args.sample_rate, max_samples=max_samples, sources=args.sources, target=args.sources)
    
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    
    config_control, config_unet = config['control'], config['unet']

    if config_control['backbone'] == 'dense':
        if not 'out_channels' in config_control.keys():
            config_control['out_channels'] = config_unet['channels'][1:]
        control_net = ControlDenseNet.build_from_config(config_control)
    else:
        raise ValueError("Invalid control net")
    unet = UNet2d.build_from_config(config_unet)
    model = ConditionedUNet2d(control_net=control_net, unet=unet)

    print(model)
    print("# Parameters: {}".format(model.num_parameters))
    
    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA", flush=True)
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA", flush=True)
        
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
    if args.criterion == 'l1loss':
        criterion = L1Loss(dim=(2,3), reduction='mean')
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    trainer = AdhocTrainer(model, loader, criterion, optimizer, args)
    trainer.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
