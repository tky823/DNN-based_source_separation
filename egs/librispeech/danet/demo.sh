#!/bin/bash

. ./path.sh

exp_dir="$1"

n_sources=2

sr=16000

window_fn='hamming'
fft_size=256
hop_size=64
ideal_mask='ibm'
threshold=40

# Model configuration
K=20
H=256
B=4
causal=0
mask_nonlinear='sigmoid'

# Criterion
criterion='l2loss'

# Optimizer
optimizer='rmsprop'
lr=1e-4
weight_decay=0

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111

if [ -z "${exp_dir}" ]; then
    exp_dir="./exp"
fi

save_dir="${exp_dir}/${n_sources}mix/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window_${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_H${H}_causal${causal}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}/seed${seed}"

model_choice="best"
model_path="${save_dir}/model/${model_choice}.pth"

num_chunk=256
duration=5

demo.py \
--sr ${sr} \
--window_fn ${window_fn} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--num_chunk ${num_chunk} \
--duration ${duration} \
--model_path ${model_path} \
--save_dir "./results"
