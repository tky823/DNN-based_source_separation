#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

sources="[bass,drums,other,vocals]"
patch=128
valid_duration=30

musdb18_root="../../../dataset/musdb18"
sample_rate=44100

window_fn='hann'
n_fft=1024
hop_length=768

# model
control='dense' # or 'conv'
simple_or_complex='complex'
conditioning='film'
config_path="./config/${control}_${simple_or_complex}_${conditioning}.yaml"

# Criterion
criterion='l1loss'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${sources}/sr${sample_rate}/patch${patch}/${criterion}/stft${n_fft}-${hop_length}_${window_fn}-window/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train.py \
--musdb18_root ${musdb18_root} \
--config_path "${config_path}" \
--sample_rate ${sample_rate} \
--patch_size ${patch} \
--valid_duration ${valid_duration} \
--window_fn "${window_fn}" \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--sources ${sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--max_norm ${max_norm} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/train_${time_stamp}.log"
