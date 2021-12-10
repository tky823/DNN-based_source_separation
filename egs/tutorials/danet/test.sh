#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2

wav_root="../../../dataset/LibriSpeech"
test_json_path="../../../dataset/LibriSpeech/test-clean/test-${n_sources}mix.json"

sample_rate=16000

window_fn='hann'
n_fft=512
hop_length=128
ideal_mask='ibm'
threshold=40
target_type='source'

# Model configuration
K=20
H=256
B=4
dropout=0
causal=0
mask_nonlinear='sigmoid'
iter_clustering=-1
take_log=1
take_db=0

# Criterion
criterion='se'

# Optimizer
optimizer='rmsprop'
lr=1e-4
weight_decay=0

batch_size=128
epochs=100

model_choice="best"

use_cuda=0
overwrite=0
seed=111

. ./path.sh
. parse_options.sh || exit 1

prefix=""

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/${criterion}/${target_type}/stft${n_fft}-${hop_length}_${window_fn}-window"
    save_dir="${save_dir}/${ideal_mask}_threshold${threshold}/K${K}_H${H}_B${B}_causal${causal}_dropout${dropout}_mask-${mask_nonlinear}"
    if [ ${take_log} -eq 1 ]; then
        save_dir="${save_dir}/take_log"
    elif [ ${take_db} -eq 1 ]; then
        save_dir="${save_dir}/take_db"
    else
        save_dir="${save_dir}/take_identity"
    fi
    save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}/seed${seed}"
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

export CUDA_VISIBLE_DEVICES="0"

test.py \
--wav_root ${wav_root} \
--test_json_path ${test_json_path} \
--sample_rate ${sample_rate} \
--window_fn ${window_fn} \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--target_type ${target_type} \
--iter_clustering ${iter_clustering} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
