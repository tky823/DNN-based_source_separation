#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import SpectrogramTrainDataset, SpectrogramEvalDataset, TrainDataLoader, EvalDataLoader
from driver import AdhocTrainer
from models.d3net import D3Net
from criterion.distance import MeanSquaredError

parser = argparse.ArgumentParser(description="Training of D3Net")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--fft_size', type=int, default=512, help='Window length')
parser.add_argument('--hop_size', type=int, default=None, help='Hop size')
parser.add_argument('--config_path', type=str, default='config_d3net.yaml', help='Model configuration')
parser.add_argument('--sources', type=str, default='[drums,bass,others,vocals]', help='Source names')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['mse'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default: 0.001')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 128')
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
    
    samples = args.duration
    overlap = samples / 2
    args.sources = args.sources.replace('[','').replace(']','').split(',')
    args.n_sources = len(args.sources)
    
    train_dataset = SpectrogramTrainDataset(args.musdb18_root, sr=args.sr, duration=args.duration, fft_size=args.fft_size, overlap=overlap, sources=args.sources)
    valid_dataset = SpectrogramEvalDataset(args.musdb18_root, sr=args.sr, max_duration=args.valid_duration, fft_size=args.fft_size, sources=args.sources)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)

    with open(args.config_path) as file:
        config = yaml.safe_load(file.read())
    config_model = config['d3net']
    
    model = D3Net(
        config_model['in_channels'], config_model['num_features'], config_model['growth_rate'], config_model['bottleneck_channels'], kernel_size=config_model['kernel_size'], sections=config_model['sections'], scale=config_model['scale'],
        num_d3blocks=config_model['num_d3blocks'], num_d2blocks=config_model['num_d2blocks'], depth=config_model['depth'], compressed_depth=config_model['compressed_depth'],
        growth_rate_d2block=config_model['growth_rate_d2block'], kernel_size_d2block=config_model['kernel_size_d2block'], depth_d2block=config_model['depth_d2block'],
        kernel_size_gated=config_model['kernel_size_gated']
    )
    print(model, flush=True)
    print("# Parameters: {}".format(model.num_parameters))
    
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
        criterion = MeanSquaredError(dim=(1, 2, 3, 4))
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    trainer = AdhocTrainer(model, loader, criterion, optimizer, args)
    trainer.run()
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
