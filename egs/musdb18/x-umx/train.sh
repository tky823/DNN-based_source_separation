#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

sources="[drums,bass,other,vocals]"
duration=6
valid_duration=100

musdb18_root="../../../dataset/musdb18"
sr=44100

window_fn='hann'
fft_size=4096
hop_size=1024
max_bin=1487

# Model
hidden_channels=512
num_layers=3
dropout=4e-1
causal=0

# Augmentation
augmentation_path="./config/paper/augmentation.yaml"

# Criterion
criterion='mdl' # multi-domain loss
weight_time=1e+0
weight_frequency=1e+1

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=1e-5
max_norm=0 # 0 is handled as no clipping

batch_size=16
samples_per_epoch=6400
epochs=1000

use_norbert=0
use_cuda=1
overwrite=0
num_workers=2
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/sr${sr}/${sources}/${duration}sec/${criterion}_time${weight_time}-frequency${weight_frequency}/stft${fft_size}-${hop_size}_${window_fn}-window/H${hidden_channels}_N${num_layers}_dropout${dropout}_causal${causal}"
    if [ ${samples_per_epoch} -gt 0 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs}-s${samples_per_epoch}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    fi
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
--musdb18_root "${musdb18_root}" \
--sr ${sr} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--window_fn "${window_fn}" \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--augmentation_path "${augmentation_path}" \
--max_bin ${max_bin} \
--hidden_channels ${hidden_channels} \
--num_layers ${num_layers} \
--dropout ${dropout} \
--causal ${causal} \
--sources ${sources} \
--criterion ${criterion} \
--weight_time ${weight_time} \
--weight_frequency ${weight_frequency} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--max_norm ${max_norm} \
--batch_size ${batch_size} \
--samples_per_epoch ${samples_per_epoch} \
--epochs ${epochs} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" \
--use_norbert ${use_norbert} \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--num_workers ${num_workers} \
--seed ${seed} | tee "${log_dir}/train_${time_stamp}.log"
