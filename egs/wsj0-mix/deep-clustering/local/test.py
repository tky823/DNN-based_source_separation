#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import IdealMaskSpectrogramTestDataset, IdealMaskSpectrogramTestDataLoader
from adhoc_driver import AdhocTester
from models.deep_clustering import DeepClustering
from criterion.deep_clustering import AffinityLoss
from criterion.sdr import NegSISDR
from adhoc_criterion import AffinityLossWrapper, Metrics

parser = argparse.ArgumentParser(description="Evaluation of Deep Clustering")

parser.add_argument('--test_wav_root', type=str, default=None, help='Path for test dataset ROOT directory')
parser.add_argument('--test_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tt_mix')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--ideal_mask', type=str, default='ibm', choices=['ibm', 'irm', 'wfm'], help='Ideal mask for assignment')
parser.add_argument('--threshold', type=float, default=40, help='Wight threshold. Default: 40')
parser.add_argument('--n_fft', type=int, default=256, help='Window length')
parser.add_argument('--hop_length', type=int, default=None, help='Hop size')
parser.add_argument('--iter_clustering', type=int, default=-1, help='# iterations when clustering')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='affinity', choices=['affinity'], help='Criterion')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
parser.add_argument('--model_path', type=str, default='./tmp/model/best.pth', help='Path for model')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    test_dataset = IdealMaskSpectrogramTestDataset(args.test_wav_root, args.test_list_path, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, mask_type=args.ideal_mask, threshold=args.threshold, n_sources=args.n_sources)
    print("Test dataset includes {} samples.".format(len(test_dataset)))

    args.n_bins = args.n_fft // 2 + 1
    loader = IdealMaskSpectrogramTestDataLoader(test_dataset, batch_size=1, shuffle=False)
    model = DeepClustering.build_model(args.model_path)

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

    # Criterion
    if args.criterion == 'affinity':
        criterion = AffinityLoss()
        wrapper_criterion = AffinityLossWrapper(criterion)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    metrics = OrderedDict()
    metrics['SISDR'] = NegSISDR()
    metrics = Metrics(metrics)

    if args.iter_clustering < 0:
        args.iter_clustering = None # Iterates until convergence

    tester = AdhocTester(model, loader, wrapper_criterion, metrics, args)
    tester.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
