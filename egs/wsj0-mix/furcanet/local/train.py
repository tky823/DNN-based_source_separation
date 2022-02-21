#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import WaveTrainDataset, WaveEvalDataset, TrainDataLoader, EvalDataLoader
from adhoc_driver import AdhocTrainer
from models.furcanet import FurcaNet
from criterion.sdr import NegSISDR
from criterion.pit import PIT1d

parser = argparse.ArgumentParser(description="Training of Conv-TasNet")

parser.add_argument('--train_wav_root', type=str, default=None, help='Path for training dataset ROOT directory')
parser.add_argument('--valid_wav_root', type=str, default=None, help='Path for validation dataset ROOT directory')
parser.add_argument('--train_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tr_mix')
parser.add_argument('--valid_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_cv_mix')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--kernel_size', '-L', type=int, default=16, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size//2')
parser.add_argument('--conv_hidden_channels', '-Hc', type=int, default=128, help='Hidden channels of RNN in each direction')
parser.add_argument('--rnn_hidden_channels', '-Hr', type=int, default=128, help='Hidden channels of RNN in each direction')
parser.add_argument('--num_conv_blocks', '-Bc', type=int, default=6, help='# blocks of covolutional block.')
parser.add_argument('--num_rnn_blocks', '-Br', type=int, default=6, help='# blocks of reccurent block.')
parser.add_argument('--nonlinear', type=int, default=1, help='Nonlinear function')
parser.add_argument('--norm', type=int, default=1, help='Normalization')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['sisdr'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')
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
    
    samples = int(args.sample_rate * args.duration)
    overlap = samples // 2
    max_samples = int(args.sample_rate * args.valid_duration)
    
    train_dataset = WaveTrainDataset(args.train_wav_root, args.train_list_path, samples=samples, overlap=overlap, n_sources=args.n_sources)
    valid_dataset = WaveEvalDataset(args.valid_wav_root, args.valid_list_path, max_samples=max_samples, n_sources=args.n_sources)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    if not args.enc_nonlinear:
        args.enc_nonlinear = None
    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    model = FurcaNet(
        args.conv_hidden_channels, args.rnn_hidden_channels,
        num_conv_blocks=args.num_conv_blocks, num_rnn_blocks=args.num_rnn_blocks,
        kernel_size=args.kernel_size, stride=args.stride,
        nonlinear=args.nonlinear, norm=args.norm,
        causal=args.causal,
        n_sources=args.n_sources,
    )
    print(model)
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
    if args.criterion == 'sisdr':
        criterion = NegSISDR()
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    pit_criterion = PIT1d(criterion, n_sources=args.n_sources)
    
    trainer = AdhocTrainer(model, loader, pit_criterion, optimizer, args)
    trainer.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
