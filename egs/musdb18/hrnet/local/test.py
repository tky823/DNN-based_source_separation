#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from adhoc_dataset import SpectrogramTestDataset, TestDataLoader
from adhoc_driver import SingleTargetTester
from models.hrnet import HRNet
from criterion.distance import MeanAbsoluteError, MeanSquaredError

parser = argparse.ArgumentParser(description="Evaluation of HRNet")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sample_rate', '-sr', type=int, default=16000, help='Sampling rate')
parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
parser.add_argument('--n_fft', type=int, default=1024, help='FFT length')
parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--target', type=str, default=None, choices=['bass', 'drums', 'other', 'vocals'], help='Target name')
parser.add_argument('--criterion', type=str, default='mae', choices=['mae','mse'], help='Criterion')
parser.add_argument('--model_path', type=str, default=None, help='Path to pretrained model.')
parser.add_argument('--estimates_dir', type=str, default=None, help='Estimated sources output directory')
parser.add_argument('--json_dir', type=str, default=None, help='Json directory')
parser.add_argument('--estimate_all', type=int, default=1, help='Estimates all songs. GPU is required if use_cuda=1.')
parser.add_argument('--evaluate_all', type=int, default=1, help='Evaluates all estimations. GPU is NOT required.')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)
    
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    
    test_dataset = SpectrogramTestDataset(args.musdb18_root, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, sample_rate=args.sample_rate, patch_size=args.patch_size, sources=args.sources, target=args.target)
    print("Test dataset includes {} samples.".format(len(test_dataset)))
    
    loader = TestDataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model_path = os.path.join(args.model_path)
    model = HRNet.build_model(model_path)
    
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
    if args.criterion == 'mae':
        criterion = MeanAbsoluteError(dim=(1,2,3))
        args.save_normalized = False
    elif args.criterion == 'mse':
        criterion = MeanSquaredError(dim=(1,2,3))
        args.save_normalized = False
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))
    
    tester = SingleTargetTester(model, loader, criterion, args)
    tester.run()
    
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
