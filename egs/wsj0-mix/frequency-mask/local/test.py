#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from dataset import WaveTestDataset, TestDataLoader
from criterion.sdr import NegSISDR
from adhoc_driver import AdhocTester
from adhoc_utils import IdealBinaryMasking, IdealRatioMasking, PhaseSensitiveMasking

parser = argparse.ArgumentParser(description="Evaluation of frequency masking")

parser.add_argument('--test_wav_root', type=str, default=None, help='Path for test dataset ROOT directory')
parser.add_argument('--test_list_path', type=str, default=None, help='Path for mix_<n_sources>_spk_<max,min>_tt_mix')
parser.add_argument('--sample_rate', '-sr', type=int, default=10, help='Sampling rate')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--method', type=str, default='ibm', choices=['ibm', 'irm', 'psm'], help='Ideal mask for assignment')
parser.add_argument('--n_fft', type=int, default=256, help='Window length')
parser.add_argument('--hop_length', type=int, default=None, help='Hop size')
parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
parser.add_argument('--criterion', type=str, default='sisdr', choices=['sisdr'], help='Criterion')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')

def main(args):
    test_dataset = WaveTestDataset(args.test_wav_root, args.test_list_path, n_sources=args.n_sources)
    print("Test dataset includes {} samples.".format(len(test_dataset)))

    args.n_bins = args.n_fft // 2 + 1
    loader = TestDataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.method == 'ibm':
        method = IdealBinaryMasking(args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn)
    elif args.method == 'irm':
        method = IdealRatioMasking(args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn)
    elif args.method == 'psm':
        method = PhaseSensitiveMasking(args.n_fft, hop_length=args.hop_length, window_fn=args.window_fn)
    else:
        raise NotImplementedError("Not support {}.".format(args.method))

    # Criterion
    if args.criterion == 'sisdr':
        criterion = NegSISDR()
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    tester = AdhocTester(method, loader, criterion, args)
    tester.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
