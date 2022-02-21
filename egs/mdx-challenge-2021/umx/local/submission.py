#!/usr/bin/env python
#
# This file uses Open-Unmix for music demixing.
#
import argparse

from adhoc_predictor import UMXPredictor

parser = argparse.ArgumentParser(description="Evaluation of Open-Unmix")

parser.add_argument('--sample_rate', '-sr', type=int, default=44100, help='Sampling rate.')
parser.add_argument('--duration', type=float, default=256, help='Duration')
parser.add_argument('--n_fft', type=int, default=4096, help='FFT length')
parser.add_argument('--hop_length', type=int, default=1024, help='Hop length')
parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
parser.add_argument('--sources', type=str, default="[bass,drums,other,vocals]", help='Source names')
parser.add_argument('--model_dir', type=str, default='./tmp', help='Path to model.')

def main(args):
    args.sources = args.sources.replace('[', '').replace(']', '').split(',')
    patch_samples = int(args.duration * args.sample_rate)
    args.patch_size = (patch_samples + 2 * (args.n_fft // 2) - args.n_fft) // args.hop_length - 1

    submission = UMXPredictor(args)
    submission.run()

    print("Successfully generated predictions!")

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)