#!/bin/bash

exp_dir="./exp"
continue_from=""

sources="[drums,bass,others,vocals]"
duration=4
valid_duration=10

musdb18_root="../../../dataset/musdb18"
sr=44100

window_fn='hann' # window_fn is activated if enc_bases='Fourier' or dec_bases='Fourier'
K=512

# Separator
N=8
M=3

# Criterion
criterion='sisdr'

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

. ./path.sh
. parse_options.sh || exit 1

save_dir="${exp_dir}/${sources}/sr${sr}/${duration}sec/${criterion}/K${K}_N${N}_M${M}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="0"

train.py \
--musdb18_root ${musdb18_root} \
--sr ${sr} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--window_fn "${window_fn}" \
-N ${N} \
-M ${M} \
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
