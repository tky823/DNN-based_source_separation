#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2

sample_rate=16000

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

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window_${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_causal${causal}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_choice="last"
model_path="${save_dir}/model/${model_choice}.pth"

n_sources_test=2
iter_clustering=100

num_chunk=256
duration=5

demo.py \
--sample_rate ${sample_rate} \
--window_fn ${window_fn} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--n_sources ${n_sources_test} \
--iter_clustering ${iter_clustering} \
--num_chunk ${num_chunk} \
--duration ${duration} \
--model_path "${model_path}" \
--save_dir "./results"
