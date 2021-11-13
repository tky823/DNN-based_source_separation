#!/bin/bash

exp_dir="./exp"
tag=""

sources="[bass,drums,other,vocals]"
target='vocals'
patch=256
valid_duration=100

musdb18_root="../../../dataset/MUSDB18"
sample_rate=16000

window_fn='hann'
fft_size=1024
hop_size=512

# Criterion
criterion='mae'

# Optimizer
optimizer='adam'
lr=1e-4
weight_decay=1e-5
max_norm=0 # 0 is handled as no clipping

batch_size=5
samples_per_epoch=6400
epochs=1000

use_cuda=1
seed=111
gpu_id="0"

model_choice="best" # 'last' or 'best'

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/sr${sample_rate}/${sources}/patch${patch}/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window"
    if [ ${samples_per_epoch} -gt 0 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs}-s${samples_per_epoch}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    fi
else
    save_dir="${exp_dir}/${tag}"
fi

model_path="${save_dir}/model/${target}/${model_choice}.pth"
log_dir="${save_dir}/log/test/${target}/${model_choice}"
json_dir="${save_dir}/json/${target}/${model_choice}"

musdb=`basename "${musdb18_root}"` # 'MUSDB18' or 'MUSDB18HQ'
estimates_dir="${save_dir}/${musdb}/${model_choice}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--musdb18_root ${musdb18_root} \
--sample_rate ${sample_rate} \
--patch_size ${patch} \
--window_fn "${window_fn}" \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--sources ${sources} \
--target ${target} \
--criterion ${criterion} \
--estimates_dir "${estimates_dir}" \
--json_dir "${json_dir}" \
--model_path "${model_path}" \
--estimate_all ${estimate_all} \
--evaluate_all ${evaluate_all} \
--use_cuda ${use_cuda} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"