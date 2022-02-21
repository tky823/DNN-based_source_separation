#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sample_rate=${sr_k}000
duration=0.75
valid_duration=10
max_or_min='min'

train_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tr"
valid_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/cv"

train_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tr_mix"
valid_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_cv_mix"

# Model
latent_dim=512

spk_kernel_size=3
spk_num_layers=14
sep_kernel_size_in=4
sep_kernel_size=3
sep_num_layers=10
sep_num_blocks=4

dilated=1
separable=1
causal=0
nonlinear='prelu'
norm=1

# Criterion
reconst_criterion='sdr' # or 'sisdr'
spk_criterion='distance'
reg_criterion='entropy'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=5

batch_size=2
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec"
    if [ "${reg_criterion}" = 'none' ]; then
        save_dir="${save_dir}/${reconst_criterion}-${spk_criterion}-${reg_criterion}"
    else
        save_dir="${save_dir}/${reconst_criterion}-${spk_criterion}"
    fi
    save_dir="${save_dir}/latent${latent_dim}/spk${spk_kernel_size}_${spk_num_layers}/sep${sep_kernel_size_in}-${sep_kernel_size}_${sep_num_blocks}-${sep_num_layers}"
    save_dir="${save_dir}/dilated${dilated}_separable${separable}_causal${causal}_${nonlinear}_norm${norm}"
    save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
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
--train_wav_root "${train_wav_root}" \
--valid_wav_root "${valid_wav_root}" \
--train_list_path ${train_list_path} \
--valid_list_path ${valid_list_path} \
--sample_rate ${sample_rate} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--latent_dim ${latent_dim} \
--spk_kernel_size ${spk_kernel_size} \
--spk_num_layers ${spk_num_layers} \
--sep_kernel_size_in ${sep_kernel_size_in} \
--sep_kernel_size ${sep_kernel_size} \
--sep_num_layers ${sep_num_layers} \
--sep_num_blocks ${sep_num_blocks} \
--dilated ${dilated} \
--separable ${separable} \
--causal ${causal} \
--nonlinear ${nonlinear} \
--norm ${norm} \
--n_sources ${n_sources} \
--reconst_criterion ${reconst_criterion} \
--spk_criterion ${spk_criterion} \
--reg_criterion ${reg_criterion} \
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
