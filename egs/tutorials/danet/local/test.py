#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import IdealMaskSpectrogramTestDataset, AttractorTestDataLoader
from driver import AttractorTester
from models.danet import DANet
from adhoc_criterion import SquaredError
from criterion.pit import PIT2d

parser = argparse.ArgumentParser(description="Evaluation of DANet")

parser.add_argument('--wav_root', type=str, default=None, help='Path for dataset ROOT directory')
parser.add_argument('--test_json_path', type=str, default=None, help='Path for test.json')
parser.add_argument('--sample_rate', '-sr', type=int, default=16000, help='Sampling rate')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--ideal_mask', type=str, default='ibm', choices=['ibm', 'irm', 'wfm'], help='Ideal mask for assignment')
parser.add_argument('--threshold', type=float, default=40, help='Weight threshold. Default: 40 ')
parser.add_argument('--n_fft', type=int, default=512, help='Window length')
parser.add_argument('--hop_length', type=int, default=None, help='Hop size')
parser.add_argument('--iter_clustering', type=int, default=10, help='# iterations when clustering')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='se', choices=['se'], help='Criterion')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
parser.add_argument('--model_path', type=str, default='./tmp/model/best.pth', help='Path for model')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    test_dataset = IdealMaskSpectrogramTestDataset(args.wav_root, args.test_json_path, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, mask_type=args.ideal_mask, threshold=args.threshold)
    print("Test dataset includes {} samples.".format(len(test_dataset)))

    args.n_bins = args.n_fft // 2 + 1
    loader = AttractorTestDataLoader(test_dataset, batch_size=1, shuffle=False)

    model = DANet.build_model(args.model_path)
    print(model)
    print("# Parameters: {}".format(model.num_parameters))

    if model.iter_clustering != args.iter_clustering:
        print("model.iter_clustering is changed from {} -> {}.".format(model.iter_clustering, args.iter_clustering))
        model.iter_clustering = args.iter_clustering

    if args.use_cuda:
        if torch.cuda.is_available():
            model.cuda()
            model = nn.DataParallel(model)
            print("Use CUDA", flush=True)
        else:
            raise ValueError("Cannot use CUDA.")
    else:
        print("Does NOT use CUDA", flush=True)

    # Criterion
    if args.criterion == 'se':
        criterion = SquaredError(sum_dim=2, mean_dim=(1,3)) # (batch_size, n_sources, n_bins, n_frames)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    pit_criterion = PIT2d(criterion, n_sources=args.n_sources)

    tester = AttractorTester(model, loader, pit_criterion, args)
    tester.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
