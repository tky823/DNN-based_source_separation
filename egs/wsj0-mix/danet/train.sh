#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=0.8 # 6400 samples
valid_duration=10
max_or_min='min'

train_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tr"
valid_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/cv"

train_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tr_mix"
valid_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_cv_mix"

window_fn='hann'
fft_size=256
hop_size=64
ideal_mask='ibm'
threshold=40

# Embedding dimension
K=20

# Network configuration
H=300
B=4
causal=0
mask_nonlinear='sigmoid'
iter_clustering=10

# Criterion
criterion='l2loss'

# Optimizer
optimizer='rmsprop'
lr=1e-4
lr_end=3e-6
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=64
epochs=150

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window_${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_causal${causal}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-${lr_end}-decay${weight_decay}_clip${max_norm}/seed${seed}"
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
--train_wav_root ${train_wav_root} \
--valid_wav_root ${valid_wav_root} \
--train_list_path ${train_list_path} \
--valid_list_path ${valid_list_path} \
--sr ${sr} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--window_fn "${window_fn}" \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
-K ${K} \
-H ${H} \
-B ${B} \
--causal ${causal} \
--mask_nonlinear ${mask_nonlinear} \
--iter_clustering ${iter_clustering} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--lr ${lr} \
--lr_end ${lr_end} \
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
