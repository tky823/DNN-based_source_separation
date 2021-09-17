#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=0.8 # 6400 samples
max_or_min='min'

wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

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

model_choice="best"

model_dir="${save_dir}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/log"
out_dir="${save_dir}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--test_wav_root ${wav_root} \
--test_list_path ${test_list_path} \
--sr ${sr} \
--window_fn ${window_fn} \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--iter_clustering ${iter_clustering} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
