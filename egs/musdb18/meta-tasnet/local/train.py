#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import warnings

import torch
import torch.nn as nn

from utils.utils import set_seed
from adhoc_dataset import WaveTrainDataset, WaveEvalDataset, EvalDataLoader
from dataset import TrainDataLoader
from models.meta_tasnet import MetaTasNet
from criterion.sdr import NegSISDR
from adhoc_criterion import NegSimilarityLoss, MultiDissimilarityLoss, MultiLoss
from adhoc_driver import Trainer

SAMPLE_RATE_MUSDB18 = 16000

parser = argparse.ArgumentParser(description="Training of Meta-TasNet")

parser.add_argument('--musdb18_root', type=str, default=None, help='Path to MUSDB18')
parser.add_argument('--is_wav', type=int, default=0, help='0: extension is wav (MUSDB), 1: extension is not .wav, is expected .mp4 (MUSDB-HQ)')
parser.add_argument('--sample_rate', '-sr', type=str, default='[8000,16000,32000]', help='Sampling rate')
parser.add_argument('--duration', type=float, default=2, help='Duration')
parser.add_argument('--valid_duration', type=float, default=4, help='Duration for valid dataset for avoiding memory error.')
parser.add_argument('--stage', type=int, default=1, help='Stage')
parser.add_argument('--enc_bases', type=str, default='trainable', choices=['trainable'], help='Encoder type')
parser.add_argument('--dec_bases', type=str, default='trainable', choices=['trainable'], help='Decoder type')
parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--n_bases', '-N', type=int, default=512, help='# bases')
parser.add_argument('--kernel_size', '-L', type=int, default=16, help='Kernel size')
parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size')
parser.add_argument('--sep_bottleneck_channels', '-B', type=int, default=128, help='Bottleneck channels of separator')
parser.add_argument('--sep_hidden_channels', '-H', type=int, default=128, help='Hidden channels of separator')
parser.add_argument('--sep_skip_channels', '-Sc', type=int, default=128, help='Skip connection channels of separator')
parser.add_argument('--sep_kernel_size', '-P', type=int, default=3, help='Skip connection channels of separator')
parser.add_argument('--sep_num_layers', '-X', type=int, default=8, help='# layers of separator')
parser.add_argument('--sep_num_blocks', '-R', type=int, default=3, help='# blocks of separator. Each block has R layers')
parser.add_argument('--dilated', type=int, default=1, help='Dilated convolution')
parser.add_argument('--separable', type=int, default=1, help='Depthwise-separable convolution')
parser.add_argument('--causal', type=int, default=0, help='Causality')
parser.add_argument('--sep_nonlinear', type=str, default=None, help='Non-linear function of separator')
parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
parser.add_argument('--conv_name', type=str, default='generated', help='Conv1D type')
parser.add_argument('--norm_name', type=str, default='generated', help='Normalization type')
parser.add_argument('--embed_dim', type=int, default=0, help='Source embedding dimension')
parser.add_argument('--embed_bottleneck_channels', type=int, default=0, help='Bottleneck channels in embedding module.')
parser.add_argument('--n_fft', type=int, default=None, help='# of FFT samples.')
parser.add_argument('--hop_length', type=int, default=None, help='Hop length of STFT')
parser.add_argument('--enc_compression_rate', type=int, default=4, help='Compression rate')
parser.add_argument('--num_filters', type=int, default=6, help='# of filters')
parser.add_argument('--n_mels', type=int, default=256, help='# of mel bins')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--sources', type=str, default='[bass,drums,others,vocals]', help='Source names')
parser.add_argument('--criterion_reconstruction', type=str, default='sisdr', help='Reconstruction criterion')
parser.add_argument('--criterion_similarity', type=str, default='cos', help='Simirarity criterion')
parser.add_argument('--reconstruction', type=float, default=5e-2, help='Weight for reconstrunction loss')
parser.add_argument('--similarity', type=float, default=2e+0, help='Weight for similarity loss')
parser.add_argument('--dissimilarity', type=float, default=3e+0, help='Weight for dissimilarity loss')
parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'radam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate. Default: 0.001')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size. Default: 12')
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
    
    args.sources = args.sources.replace('[','').replace(']','').split(',')
    args.n_sources = len(args.sources)
    args.sample_rate = [int(sample_rate) for sample_rate in args.sample_rate.replace('[', '').replace(']', '').split(',')]
    samples_per_epoch = int(40 * 3000 // args.duration)
    
    train_dataset = WaveTrainDataset(args.musdb18_root, samples_per_epoch=samples_per_epoch, sources=args.sources, is_wav=args.is_wav)
    valid_dataset = WaveEvalDataset(args.musdb18_root, max_duration=args.valid_duration, sources=args.sources, is_wav=args.is_wav)
    print("Training dataset includes {} samples.".format(len(train_dataset)))
    print("Valid dataset includes {} samples.".format(len(valid_dataset)))
    
    loader = {}
    loader['train'] = TrainDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    loader['valid'] = EvalDataLoader(valid_dataset, batch_size=1, shuffle=False)
    
    if not args.enc_nonlinear:
        args.enc_nonlinear = None

    if args.conv_name == 'generated' or args.norm_name == 'generated':
        if args.embed_dim <= 0:
            raise ValueError("Invalid `embed_dim`.")
        if args.embed_bottleneck_channels <= 0:
            raise ValueError("Invalid `embed_bottleneck_channels`.")
        kwargs = {
            'embed_dim': args.embed_dim,
            'embed_bottleneck_channels': args.embed_bottleneck_channels,
        }
    else:
        if args.embed_dim > 0:
            warnings.warn("`embed_dim` is NOT used.", UserWarning)
        if args.embed_bottleneck_channels > 0:
            warnings.warn("`embed_bottleneck_channels` is NOT used.", UserWarning)
        kwargs = {}
    if args.n_fft is None:
        args.n_fft = 1024 * (args.sample_rate[0] // 8000)
    if args.hop_length is None:
        args.hop_length = args.n_fft // 4
    
    model = MetaTasNet(
        args.n_bases, args.kernel_size, stride=args.stride,
        enc_n_fft=args.n_fft, enc_hop_length=args.hop_length, enc_compression_rate=args.enc_compression_rate,
        num_filters=args.num_filters, n_mels=args.n_mels,
        sep_hidden_channels=args.sep_hidden_channels, sep_bottleneck_channels=args.sep_bottleneck_channels, sep_skip_channels=args.sep_skip_channels,
        sep_kernel_size=args.sep_kernel_size, sep_num_blocks=args.sep_num_blocks, sep_num_layers=args.sep_num_layers,
        dilated=args.dilated, separable=args.separable, dropout=args.dropout,
        sep_nonlinear=args.sep_nonlinear, mask_nonlinear=args.mask_nonlinear,
        causal=args.causal,
        conv_name=args.conv_name, norm_name=args.norm_name,
        num_stages=len(args.sample_rate), n_sources=args.n_sources,
        **kwargs
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
    elif args.optimizer == 'radam':
        from adhoc_optimizer import Ranger
        optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Not support optimizer {}".format(args.optimizer))
    
    # Criterion
    metrics, weights = {}, {}

    if args.criterion_reconstruction == 'sisdr':
        metrics['main'], metrics['reconstruction'] = NegSISDR(), NegSISDR()
        weights['main'], weights['reconstruction'] = 1, args.reconstruction
    else:
        raise ValueError("Not support criterion {}".format(args.criterion_reconstruction))
    
    if args.criterion_similarity == 'cos':
        metrics['similarity'], metrics['dissimilarity'] = NegSimilarityLoss(), MultiDissimilarityLoss(n_sources=args.n_sources)
        weights['similarity'], weights['dissimilarity'] = args.similarity, args.dissimilarity
    else:
        raise ValueError("Not support criterion {}".format(args.criterion_similarity))
    
    criterion = MultiLoss(metrics, weights)

    if args.max_norm is not None and args.max_norm == 0:
        args.max_norm = None
    
    trainer = Trainer(model, loader, criterion, optimizer, args)
    trainer.run()
  
if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)