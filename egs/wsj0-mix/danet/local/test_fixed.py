#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from utils.utils import set_seed
from dataset import IdealMaskSpectrogramTrainDataset, IdealMaskSpectrogramTestDataset, TrainDataLoader, AttractorTestDataLoader
from adhoc_data_parallel import AdhocDataParallel
from adhoc_driver import FixedAttractorComputer, FixedAttractorTester
from models.danet import DANet, FixedAttractorDANet
from criterion.pit import PIT2d
from criterion.sdr import NegSISDR
from adhoc_criterion import Metrics, SquaredError

parser = argparse.ArgumentParser(description="Evaluation of DANet using fixed attractor.")

parser.add_argument('--test_wav_root', type=str, default=None, help='Path for test dataset ROOT directory')
parser.add_argument('--test_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tt_mix')
parser.add_argument('--sample_rate', '-sr', type=int, default=8000, help='Sampling rate')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--ideal_mask', type=str, default='ibm', choices=['ibm', 'irm', 'wfm'], help='Ideal mask for assignment')
parser.add_argument('--threshold', type=float, default=40, help='Wight threshold. Default: 40')
parser.add_argument('--target_type', type=str, default='source', choices=['source', 'oracle'], help='Target type DNN tries to output.')
parser.add_argument('--n_fft', type=int, default=256, help='Window length')
parser.add_argument('--hop_length', type=int, default=None, help='Hop size')
parser.add_argument('--iter_clustering', type=int, default=-1, help='# iterations when clustering')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='se', choices=['se'], help='Criterion')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
parser.add_argument('--base_model_path', type=str, default='./tmp/model/best.pth', help='Path for model')
parser.add_argument('--wrapper_model_dir', type=str, default='./tmp/wrapper_model', help='Path for wrapper model')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--compute_attractor', type=int, default=1, help='0: Does not compute attractor, 1: Computes attractor')
parser.add_argument('--estimate_all', type=int, default=1, help='0: Does not estimate, 1: Estimates all data')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    if args.iter_clustering < 0:
        args.iter_clustering = None # Iterates until convergence

    if args.compute_attractor:
        test_dataset = IdealMaskSpectrogramTrainDataset(args.test_wav_root, args.test_list_path, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, mask_type=args.ideal_mask, threshold=args.threshold, n_sources=args.n_sources)
        print("Test dataset includes {} samples.".format(len(test_dataset)))

        args.n_bins = args.n_fft // 2 + 1
        loader = TrainDataLoader(test_dataset, batch_size=1, shuffle=False)

        model = DANet.build_model(args.base_model_path)
        print(model)
        print("# Parameters: {}".format(model.num_parameters))

        if args.use_cuda:
            if torch.cuda.is_available():
                model.cuda()
                model = AdhocDataParallel(model)
                print("Use CUDA", flush=True)
            else:
                raise ValueError("Cannot use CUDA.")
        else:
            print("Does NOT use CUDA", flush=True)

        args.wrapper_class = FixedAttractorDANet

        computer = FixedAttractorComputer(model, loader, args)
        computer.run()

    if args.estimate_all:
        test_dataset = IdealMaskSpectrogramTestDataset(args.test_wav_root, args.test_list_path, n_fft=args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn, mask_type=args.ideal_mask, threshold=args.threshold, n_sources=args.n_sources)
        print("Test dataset includes {} samples.".format(len(test_dataset)))

        args.n_bins = args.n_fft // 2 + 1
        loader = AttractorTestDataLoader(test_dataset, batch_size=1, shuffle=False)

        base_model_filename = os.path.basename(args.base_model_path)
        args.model_path = os.path.join(args.wrapper_model_dir, base_model_filename)
        model = FixedAttractorDANet.build_model(args.model_path, load_state_dict=True)
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
        if args.criterion == 'se':
            criterion = SquaredError(sum_dim=2, mean_dim=(1,3)) # (batch_size, n_sources, n_bins, n_frames)
        else:
            raise ValueError("Not support criterion {}".format(args.criterion))

        pit_criterion = PIT2d(criterion, n_sources=args.n_sources)

        metrics = OrderedDict()
        metrics['SISDR'] = NegSISDR()
        metrics = Metrics(metrics)

        tester = FixedAttractorTester(model, loader, pit_criterion, metrics, args)
        tester.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
