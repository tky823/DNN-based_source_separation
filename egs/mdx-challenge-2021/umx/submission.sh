#!/bin/bash

. ./path.sh

n_fft=4096
hop_length=1024
window_fn='hann'

sample_rate=44100
duration=6

model_choice='best'
model_dir="./pretrained/paper-musdb18/${model_choice}" # `model_dir` must includes "bass.pth", "drums.pth", "other.pth", and "vocals.pth".

submission.py \
--sample_rate ${sample_rate} \
--duration ${duration} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--window_fn ${window_fn} \
--model_dir ${model_dir}