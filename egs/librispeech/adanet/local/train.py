#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import IdealMaskSpectrogramTrainDataset, IdealMaskSpectrogramEvalDataset, TrainDataLoader, EvalDataLoader
from driver import AttractorTrainer
from models.adanet import ADANet
from criterion.distance import L2Loss

parser = argparse.ArgumentParser(description="Training of ADANet (Anchored Deep Attractor Network)")

parser.add_argument('--wav_root', type=str, default=None, help='Path for dataset ROOT directory')
parser.add_argument('--train_json_path', type=str, default=None, help='Path for train.json')
parser.add_argument('--valid_json_path', type=str, default=None, help='Path for valid.json')
parser.add_argument('--sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--ideal_mask', type=str, default='ibm', choices=['ibm', 'irm', 'wfm'], help='Ideal mask for assignment')
parser.add_argument('--threshold', type=float, default=40, help='Wight threshold. Default: 40 ')

# Model configuration
parser.add_argument('--fft_size', type=int, default=256, help='Window length')
parser.add_argument('--hop_size', type=int, default=None, help='Hop size')
parser.add_argument('--embed_dim', '-K', type=int, default=20, help='Embedding dimension')
parser.add_argument('--hidden_channels', '-H', type=int, default=600, help='hidden_channels')
parser.add_argument('--num_blocks', '-B', type=int, default=4, help='# LSTM layers')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['sisdr'], help='Criterion')
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
    
    loader = {}
    
    train_dataset = SpectrogramTrainDataset(args.wav_root, args.train_json_path)
    valid_dataset = SpectrogramTrainDataset(args.wav_root, args.valid_json_path)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = TrainDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    for mixture, sources in loader['train']:
        print(mixture.size(), sources.size())
        raise ValueError("Stop")
    
    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA")
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA")
    
