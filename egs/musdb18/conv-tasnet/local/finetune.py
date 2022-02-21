#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import yaml
import torch
import torch.nn as nn

from utils.utils import set_seed
from utils.augmentation import SequentialAugmentation, choose_augmentation
from dataset import AugmentationWaveTrainDataset, TrainDataLoader, EvalDataLoader
from adhoc_dataset import WaveEvalDataset
from adhoc_driver import AdhocFinetuneTrainer
from models.conv_tasnet import ConvTasNet
from criterion.distance import MeanAbsoluteError, MeanSquaredError
from criterion.sdr import NegSDR, NegSISDR

parser = argparse.ArgumentParser(description="Training of Conv-TasNet")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--sample_rate', type=int, default=44100, help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--augmentation_path', type=str, default=None, help='Path to augmentation.yaml')
parser.add_argument('--enc_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase'], help='Encoder type')
parser.add_argument('--dec_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase', 'pinv'], help='Decoder type')
parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--enc_onesided', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns kernel_size // 2 + 1 bins.')
parser.add_argument('--enc_return_complex', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns complex tensor, otherwise real tensor concatenated real and imaginary part in feature dimension.')
parser.add_argument('--n_basis', '-N', type=int, default=256, help='# basis')
parser.add_argument('--kernel_size', '-L', type=int, default=20, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size//2')
parser.add_argument('--sep_bottleneck_channels', '-B', type=int, default=256, help='Bottleneck channels of separator')
parser.add_argument('--sep_hidden_channels', '-H', type=int, default=512, help='Hidden channels of separator')
parser.add_argument('--sep_skip_channels', '-Sc', type=int, default=128, help='Skip connection channels of separator')
parser.add_argument('--sep_kernel_size', '-P', type=int, default=3, help='Skip connection channels of separator')
parser.add_argument('--sep_num_layers', '-X', type=int, default=10, help='# layers of separator')
parser.add_argument('--sep_num_blocks', '-R', type=int, default=4, help='# blocks of separator. Each block has R layers')
parser.add_argument('--dilated', type=int, default=1, help='Dilated convolution')
parser.add_argument('--separable', type=int, default=1, help='Depthwise-separable convolution')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--sep_nonlinear', type=str, default=None, help='Non-linear function of separator')
parser.add_argument('--sep_norm', type=int, default=1, help='Normalization')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--sources', type=str, default='[bass,drums,others,vocals]', help='Source names')
parser.add_argument('--criterion', type=str, default='mse', choices=['mae', 'mse', 'sisdr', 'sdr'], help='Criterion')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate. Default: 3e-4')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 128')
parser.add_argument('--samples_per_epoch', type=int, default=-1, help='Training samples in one epoch')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
parser.add_argument('--overwrite', type=int, default=0, help='0: NOT overwrite, 1: FORCE overwrite')
parser.add_argument('--num_workers', type=int, default=0, help='# of workers given to data loader for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

def main(args):
    set_seed(args.seed)

    args.sources = args.sources.replace('[','').replace(']','').split(',')
    args.n_sources = len(args.sources)

    if args.samples_per_epoch <= 0:
        args.samples_per_epoch = None

    with open(args.augmentation_path) as f:
        config_augmentation = yaml.safe_load(f)

    augmentation = SequentialAugmentation()
    for name in config_augmentation['augmentation']:
        augmentation.append(choose_augmentation(name, **config_augmentation[name]))

    train_dataset = AugmentationWaveTrainDataset(args.musdb18_root, sample_rate=args.sample_rate, duration=args.duration, samples_per_epoch=args.samples_per_epoch, sources=args.sources, target=args.sources, augmentation=augmentation)
    valid_dataset = WaveEvalDataset(args.musdb18_root, sample_rate=args.sample_rate, max_duration=args.valid_duration, sources=args.sources)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))

    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)

    if not args.stride:
        args.stride = args.kernel_size // 2

    if not args.enc_nonlinear:
        args.enc_nonlinear = None

    model = ConvTasNet(
        args.n_basis, args.kernel_size, stride=args.stride, in_channels=2, enc_basis=args.enc_basis, dec_basis=args.dec_basis, enc_nonlinear=args.enc_nonlinear,
        window_fn=args.window_fn, enc_onesided=args.enc_onesided, enc_return_complex=args.enc_return_complex,
        sep_hidden_channels=args.sep_hidden_channels, sep_bottleneck_channels=args.sep_bottleneck_channels, sep_skip_channels=args.sep_skip_channels, sep_kernel_size=args.sep_kernel_size, sep_num_blocks=args.sep_num_blocks, sep_num_layers=args.sep_num_layers,
        dilated=args.dilated, separable=args.separable, sep_nonlinear=args.sep_nonlinear, sep_norm=args.sep_norm, mask_nonlinear=args.mask_nonlinear,
        causal=args.causal,
        n_sources=args.n_sources
    )
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

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None

    trainer = AdhocFinetuneTrainer(model, loader, criterion, optimizer, args)
    trainer.run()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
