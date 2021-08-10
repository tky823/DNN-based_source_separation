#!/bin/bash

exp_dir="./exp"

sources="[drums,bass,other,vocals]"
target='vocals'
patch=256

musdb18_root="../../../dataset/musdb18"
is_wav=0 # If MUSDB is used, select 0. If MUSDB-HQ is used select 1.
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
epochs=50
anneal_epoch=40

use_cuda=1
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

save_dir="${exp_dir}/sr${sr}/${sources}/patch${patch}/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window/b${batch_size}_e${epochs}-${anneal_epoch}_${optimizer}-lr${lr}-${anneal_lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_choice="best"

model_dir="${save_dir}/${target}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/${target}/log"

if [ ${is_wav} -eq 0 ]; then
    out_dir="${save_dir}/musdb18/test"
else
    out_dir="${save_dir}/musdb18hq/test"
fi

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--musdb18_root ${musdb18_root} \
--is_wav ${is_wav} \
--sr ${sr} \
--patch_size ${patch} \
--window_fn "${window_fn}" \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--sources ${sources} \
--target ${target} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
