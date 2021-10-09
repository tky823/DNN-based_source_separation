#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2

wav_root="../../../dataset/LibriSpeech"
test_json_path="../../../dataset/LibriSpeech/test-clean/test-${n_sources}mix.json"

sr=16000

window_fn='hamming'
fft_size=256
hop_size=64
ideal_mask='ibm'
threshold=40

# Model configuration
K=20
H=256
B=4
causal=0
mask_nonlinear='sigmoid'
iter_clustering=10

# Criterion
criterion='l2loss'

# Optimizer
optimizer='rmsprop'
lr=1e-4
weight_decay=0

batch_size=128
epochs=100

use_cuda=0
overwrite=0
seed=111

. ./path.sh
. parse_options.sh || exit 1

prefix=""

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window_${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_causal${causal}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_choice="best"

model_dir="${save_dir}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/log"
out_dir="${save_dir}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="0"

test.py \
--wav_root ${wav_root} \
--test_json_path ${test_json_path} \
--sr ${sr} \
--window_fn ${window_fn} \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--iter_clustering ${iter_clustering} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
