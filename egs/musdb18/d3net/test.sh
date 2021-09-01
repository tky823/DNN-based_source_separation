#!/bin/bash

exp_dir="./exp"
tag=""

sources="[drums,bass,other,vocals]"
patch=256

musdb18_root="../../../dataset/musdb18"
sr=44100

window_fn='hann'
fft_size=4096
hop_size=1024

# Criterion
criterion='mse'

# Optimizer
optimizer='adam'
lr=1e-3
anneal_lr=1e-4
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=6
samples_per_epoch=7726
epochs=50
anneal_epoch=40

estimate_all=1
evaluate_all=1

use_norbert=0
use_cuda=1
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/sr${sr}/${sources}/patch${patch}/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window/b${batch_size}_e${epochs}-${anneal_epoch}-s${samples_per_epoch}_${optimizer}-lr${lr}-${anneal_lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_choice="last"

model_dir="${save_dir}/model"
log_dir="${save_dir}/log/test"
json_dir="${save_dir}/json"

musdb=`basename "${musdb18_root}"`
estimates_dir="${save_dir}/${musdb}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--musdb18_root ${musdb18_root} \
--sr ${sr} \
--patch_size ${patch} \
--window_fn "${window_fn}" \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--sources ${sources} \
--criterion ${criterion} \
--estimates_dir "${estimates_dir}" \
--json_dir "${json_dir}" \
--model_dir "${model_dir}" \
--model_choice "${model_choice}" \
--estimate_all ${estimate_all} \
--evaluate_all ${evaluate_all} \
--use_norbert ${use_norbert} \
--use_cuda ${use_cuda} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
