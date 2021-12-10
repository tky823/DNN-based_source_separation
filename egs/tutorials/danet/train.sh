#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

n_sources=2

wav_root="../../../dataset/LibriSpeech"
train_json_path="../../../dataset/LibriSpeech/train-clean-100/train-100-${n_sources}mix.json"
valid_json_path="../../../dataset/LibriSpeech/dev-clean/valid-${n_sources}mix.json"

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

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="0"

train.py \
--wav_root ${wav_root} \
--train_json_path ${train_json_path} \
--valid_json_path ${valid_json_path} \
--sample_rate ${sample_rate} \
--window_fn ${window_fn} \
--ideal_mask ${ideal_mask} \
--threshold ${threshold} \
--target_type ${target_type} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
-K ${K} \
-H ${H} \
-B ${B} \
--dropout ${dropout} \
--causal ${causal} \
--mask_nonlinear ${mask_nonlinear} \
--iter_clustering ${iter_clustering} \
--take_log ${take_log} \
--take_db ${take_db} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/train_${time_stamp}.log"

