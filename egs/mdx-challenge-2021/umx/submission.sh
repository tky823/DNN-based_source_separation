#!/bin/bash

. ./path.sh

fft_size=4096
hop_size=1024
window_fn='hann'

sample_rate=44100
duration=6

model_choice='best'
model_dir="./pretrained/paper-musdb18/${model_choice}" # `model_dir` must includes "bass.pth", "drums.pth", "other.pth", and "vocals.pth".

submission.py \
--sample_rate ${sample_rate} \
--duration ${duration} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--window_fn ${window_fn} \
--model_dir ${model_dir}