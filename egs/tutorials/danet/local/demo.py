#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pyaudio
import torch

from utils.utils_audio import write_wav
from algorithm.stft import BatchSTFT, BatchInvSTFT
from models.danet import DANet

parser = argparse.ArgumentParser(description="Demonstration of Conv-TasNet")

parser.add_argument('--sr', type=int, default=16000, help='Sampling rate')
parser.add_argument('--window_fn', type=str, default='hamming', help='Window function')
parser.add_argument('--fft_size', type=int, default=256, help='Window length')
parser.add_argument('--hop_size', type=int, default=None, help='Hop size')
parser.add_argument('--n_sources', type=int, default=2, help='Number of speakers')
parser.add_argument('--iter_clustering', type=int, default=10, help='Number of iterations when running clustering using Kmeans algorithm.')
parser.add_argument('--num_chunk', type=int, default=256, help='Number of chunks')
parser.add_argument('--duration', type=int, default=10, help='Duration [sec]')
parser.add_argument('--model_path', type=str, default='./best.pth', help='Path for model')
parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save estimation.')

FORMAT = pyaudio.paInt16
NUM_CHANNEL = 1
DEVICE_INDEX = 0

def main(args):
    process_offline(args.sr, args.num_chunk, duration=args.duration, model_path=args.model_path, save_dir=args.save_dir, args=args)

def process_offline(sr, num_chunk, duration=5, model_path=None, save_dir="results", args=None):
    num_loop = int(duration * sr / num_chunk)
    sequence = []
    
    P = pyaudio.PyAudio()
    
    # Record
    stream = P.open(format=FORMAT, channels=NUM_CHANNEL, rate=sr, input_device_index=DEVICE_INDEX, frames_per_buffer=num_chunk, input=True, output=False)
    
    for i in range(num_loop):
        input = stream.read(num_chunk)
        sequence.append(input)
        time = int(i*num_chunk/sr)
        show_progress_bar(time, duration)
    
    show_progress_bar(duration, duration)
    print()
    
    stream.stop_stream()
    stream.close()
    P.terminate()
    
    print("Stop recording")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save
    signal = b"".join(sequence)
    signal = np.frombuffer(signal, dtype=np.int16)
    signal = signal / 32768
    
    save_path = os.path.join(save_dir, "mixture.wav")
    write_wav(save_path, signal=signal, sr=sr)

    # Separate by DNN
    model = load_model(model_path)
    model.eval()
    
    fft_size, hop_size = args.fft_size, args.hop_size
    window_fn = args.window_fn
    
    if hop_size is None:
        hop_size = fft_size//2
    
    n_sources = args.n_sources
    iter_clustering = args.iter_clustering
    
    F_bin = fft_size//2 + 1
    stft = BatchSTFT(fft_size, hop_size=hop_size, window_fn=window_fn)
    istft = BatchInvSTFT(fft_size, hop_size=hop_size, window_fn=window_fn)

    print("Start separation...")
    
    with torch.no_grad():
        mixture = torch.Tensor(signal).float()
        T = mixture.size(0)
        mixture = mixture.unsqueeze(dim=0)
        mixture = stft(mixture).unsqueeze(dim=0)
        real, imag = mixture[:,:,:F_bin], mixture[:,:,F_bin:]
        mixture_amplitude = torch.sqrt(real**2+imag**2)
        estimated_sources_amplitude = model(mixture_amplitude, n_sources=n_sources, iter_clustering=iter_clustering) # TODO: Args, threshold
        ratio = estimated_sources_amplitude / mixture_amplitude
        real, imag = ratio * real, ratio * imag
        estimated_sources = torch.cat([real, imag], dim=2)
        estimated_sources = estimated_sources.squeeze(dim=0)
        estimated_sources = istft(estimated_sources, T=T).numpy()
    
    print("Finished separation...")
    
    for idx, estimated_source in enumerate(estimated_sources):
        save_path = os.path.join(save_dir, "estimated-{}.wav".format(idx))
        write_wav(save_path, signal=estimated_source, sr=sr)
    
def show_progress_bar(time, duration):
    rest = duration-time
    progress_bar = ">"*time + "-"*rest
    print("\rNow recording...", progress_bar, "{:2d}[sec]".format(rest), end="")

    
def load_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    model = DANet.build_model(model_path)
    model.load_state_dict(package['state_dict'])
    
    print("# Parameters: {}".format(model.num_parameters))

    return model
    

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)
    main(args)


