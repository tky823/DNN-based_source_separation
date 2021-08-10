#!/bin/bash

exp_dir="./exp"

sources="[drums,bass,other,vocals]"
patch=256

musdb18_root="../../../dataset/musdb18"
estimated_musdb18_root=""
is_wav=0
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

seed=111

. ./path.sh
. parse_options.sh || exit 1

save_dir="${exp_dir}/sr${sr}/${sources}/patch${patch}/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window/b${batch_size}_e${epochs}-${anneal_epoch}_${optimizer}-lr${lr}-${anneal_lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

if [ -z "${estimated_musdb18_root}" ]; then
    estimated_musdb18_root="${save_dir}/musdb18"
fi

json_dir="${save_dir}/eval/json"
log_dir="${save_dir}/eval/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

eval_all.py \
--musdb18_root "${musdb18_root}" \
--estimated_musdb18_root "${estimated_musdb18_root}" \
--is_wav ${is_wav} \
--json_dir "${json_dir}" \
--seed ${seed} | tee "${log_dir}/eval_${time_stamp}.log"
