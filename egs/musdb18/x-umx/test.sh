#!/bin/bash

exp_dir="./exp"
tag=""

sources="[drums,bass,other,vocals]"
duration=6

musdb18_root="../../../dataset/musdb18"
sr=44100

window_fn='hann'
fft_size=4096
hop_size=1024
max_bin=1487

# model
hidden_channels=512
num_layers=3
dropout=4e-1
causal=0

# Criterion
criterion='mse'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=1e-5
max_norm=0 # 0 is handled as no clipping

batch_size=16
samples_per_epoch=6400
epochs=1000

estimate_all=1
evaluate_all=1

use_norbert=0
use_cuda=1
seed=111
gpu_id="0"

model_choice="best" # 'last' or 'best'

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/sr${sr}/${sources}/${duration}sec/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window/H${hidden_channels}_N${num_layers}_dropout${dropout}_causal${causal}"
    if [ ${samples_per_epoch} -gt 0 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs}-s${samples_per_epoch}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    fi
else
    save_dir="${exp_dir}/${tag}"
fi

model_dir="${save_dir}/model"
log_dir="${save_dir}/log/test/${model_choice}"
json_dir="${save_dir}/json/${model_choice}"
model_path="${model_dir}/${model_choice}.pth"

musdb=`basename "${musdb18_root}"`
estimates_dir="${save_dir}/${musdb}/${model_choice}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--musdb18_root "${musdb18_root}" \
--sr ${sr} \
--duration ${duration} \
--window_fn "${window_fn}" \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--sources ${sources} \
--criterion ${criterion} \
--estimates_dir "${estimates_dir}" \
--json_dir "${json_dir}" \
--model_path "${model_path}" \
--estimate_all ${estimate_all} \
--evaluate_all ${evaluate_all} \
--use_norbert ${use_norbert} \
--use_cuda ${use_cuda} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
