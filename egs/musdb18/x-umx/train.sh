#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

sources="[bass,drums,other,vocals]"
duration=6
valid_duration=100

musdb18_root="../../../dataset/MUSDB18"
sample_rate=44100

window_fn='hann'
n_fft=4096
hop_length=1024
max_bin=1487

# Model
hidden_channels=512
num_layers=3
dropout=4e-1
causal=0
bridge=1

# Augmentation
augmentation_path="./config/paper/augmentation.yaml"

# Criterion
combination=1
criterion_time='wsdr' # time domain loss
criterion_frequency='mse' # time-frequency domain loss
weight_time=1e+0
weight_frequency=1e+0
min_pair=1
max_pair=3 # len(sources) - 1

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=1e-5
max_norm=5 # 0 is handled as no clipping
scheduler_path="./config/paper/scheduler.yaml"

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
    save_dir="${exp_dir}/sr${sample_rate}/${sources}/${duration}sec"
    if [ ${combination} -eq 1 ]; then
        save_dir="${save_dir}/comb${combination}_${min_pair}-${max_pair}"
    else
        save_dir="${save_dir}/comb${combination}"
    fi
    save_dir="${save_dir}/${criterion_time}${weight_time}-${criterion_frequency}${weight_frequency}/stft${n_fft}-${hop_length}_${window_fn}-window/bridge${bridge}/H${hidden_channels}_N${num_layers}_dropout${dropout}_causal${causal}"
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
config_dir="${save_dir}/config"
log_dir="${save_dir}/log"

if [ ! -e "${config_dir}" ]; then
    mkdir -p "${config_dir}"
fi

augmentation_dir=`dirname ${augmentation_path}`
augmentation_name=`basename ${augmentation_path}`

if [ ! -e "${config_dir}/${augmentation_name}" ]; then
    cp "${augmentation_path}" "${config_dir}/${augmentation_name}"
fi

scheduler_dir=`dirname ${scheduler_path}`
scheduler_name=`basename ${scheduler_path}`

if [ ! -e "${config_dir}/${scheduler_name}" ]; then
    cp "${scheduler_path}" "${config_dir}/${scheduler_name}"
fi

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train.py \
--musdb18_root "${musdb18_root}" \
--sample_rate ${sample_rate} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--window_fn "${window_fn}" \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--augmentation_path "${augmentation_path}" \
--max_bin ${max_bin} \
--hidden_channels ${hidden_channels} \
--num_layers ${num_layers} \
--dropout ${dropout} \
--causal ${causal} \
--bridge ${bridge} \
--sources "${sources}" \
--combination ${combination} \
--criterion_time ${criterion_time} \
--criterion_frequency ${criterion_frequency} \
--weight_time ${weight_time} \
--weight_frequency ${weight_frequency} \
--min_pair ${min_pair} \
--max_pair ${max_pair} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--max_norm ${max_norm} \
--scheduler_path "${scheduler_path}" \
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
