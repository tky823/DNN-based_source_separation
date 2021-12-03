#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sample_rate=${sr_k}000
duration=3.2 # 25600 samples
valid_duration=10
max_or_min='min'

train_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tr"
valid_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/cv"

train_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tr_mix"
valid_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_cv_mix"

window_fn='hann'
n_fft=256
hop_length=64
ideal_mask='wfm'
threshold=40
target_type='oracle'

# Embedding dimension
K=20

# Network configuration
H=300
B=4
N=6
dropout=5e-1
causal=0
mask_nonlinear='sigmoid'
take_log=1
take_db=0

# Criterion
criterion='se' # or 'l2loss'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=0 # 0 is handled as no clipping

batch_size=64
epochs=150

model_choice="last"

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${criterion}/${target_type}"
    save_dir="${save_dir}/stft${n_fft}-${hop_length}_${window_fn}-window/${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_N${N}_dropout${dropout}_causal${causal}_mask-${mask_nonlinear}"
    if [ ${take_log} -eq 1 ]; then
        save_dir="${save_dir}/take_log"
    elif [ ${take_db} -eq 1 ]; then
        save_dir="${save_dir}/take_db"
    else
        save_dir="${save_dir}/take_identity"
    fi
    save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

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
--test_wav_root "${test_wav_root}" \
--test_list_path "${test_list_path}" \
--sample_rate ${sample_rate} \
--window_fn ${window_fn} \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--target_type ${target_type} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"