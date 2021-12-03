#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2

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

batch_size=4
epochs=100

n_sources_test=2
num_chunk=256
duration=5
model_choice="last"

use_cuda=1
overwrite=0
seed=111

. ./path.sh
. parse_options.sh || exit 1

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

model_path="${save_dir}/model/${model_choice}.pth"

demo.py \
--sample_rate ${sample_rate} \
--window_fn ${window_fn} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--threshold ${threshold} \
--n_sources ${n_sources_test} \
--iter_clustering ${iter_clustering} \
--num_chunk ${num_chunk} \
--duration ${duration} \
--model_path "${model_path}" \
--save_dir "./results"
