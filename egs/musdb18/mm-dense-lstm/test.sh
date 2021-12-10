#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

sources="[bass,drums,other,vocals]"
target='vocals'
patch=256
valid_duration=100

musdb18_root="../../../dataset/MUSDB18"
sample_rate=44100

window_fn='hann'
n_fft=4096
hop_length=1024

# Augmentation
augmentation_path="./config/paper/augmentation.yaml"

# Criterion
criterion='mse'

# Optimizer
optimizer='rmsprop'
lr=1e-3
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=8
samples_per_epoch=6400 # If you specified samples_per_epoch=-1, samples_per_epoch is computed as 3863, which corresponds to total duration of training data.
epochs=50

estimate_all=1
evaluate_all=1

use_norbert=0
use_cuda=1
seed=111
gpu_id="0"

model_choice="best" # 'last' or 'best'

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/sr${sample_rate}/${sources}/patch${patch}/${criterion}/stft${n_fft}-${hop_length}_${window_fn}-window"
    if [ ${samples_per_epoch} -gt 0 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs}-s${samples_per_epoch}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    fi
else
    save_dir="${exp_dir}/${tag}"
fi

model_dir="${save_dir}/model"
log_dir="${save_dir}/log/test/${model_choice}"
json_dir="${save_dir}/json/${model_choice}"

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
--n_fft ${n_fft} \
--hop_length ${hop_length} \
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
