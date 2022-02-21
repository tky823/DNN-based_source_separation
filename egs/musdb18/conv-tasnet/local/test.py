#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from adhoc_dataset import WaveTestDataset, TestDataLoader
from adhoc_driver import AdhocTester
from models.conv_tasnet import ConvTasNet
from criterion.distance import MeanAbsoluteError, MeanSquaredError
from criterion.sdr import NegSDR, NegSISDR

parser = argparse.ArgumentParser(description="Evaluation of Conv-TasNet")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sample_rate', '-sr', type=int, default=44100, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--criterion', type=str, default='mse', choices=['mae', 'mse', 'sisdr', 'sdr'], help='Criterion')
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
    args.n_sources = len(args.sources)

    test_dataset = WaveTestDataset(args.musdb18_root, sample_rate=args.sample_rate, duration=args.duration, sources=args.sources, target=args.sources)
    print("Test dataset includes {} samples.".format(len(test_dataset)))

    loader = TestDataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ConvTasNet.build_model(args.model_path)

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

    # Criterion
    if args.criterion == 'mae':
        criterion = MeanAbsoluteError(dim=-1, reduction='mean')
        args.save_normalized = False
    elif args.criterion == 'mse':
        criterion = MeanSquaredError(dim=-1, reduction='mean')
        args.save_normalized = False
    elif args.criterion == 'sisdr':
        criterion = NegSISDR()
        args.save_normalized = True
    elif args.criterion == 'sdr':
        criterion = NegSDR()
        args.save_normalized = False
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    tester = AdhocTester(model, loader, criterion, args)
    tester.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
