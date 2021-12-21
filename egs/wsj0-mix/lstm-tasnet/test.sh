#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sample_rate=${sr_k}000
duration=4
valid_duration=10
max_or_min='min'

test_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

# Encoder & decoder
enc_basis='trainableGated' # choose from ['trainable', 'trainableGated']
dec_basis='trainable' # choose from ['trainable']
enc_nonlinear='' # enc_nonlinear is activated if enc_basis='trainable'

N=500
L=40

# Separator
H=500
X=2
R=2 # R x X is actual number of layers in LSTM.
causal=0
mask_nonlinear='sigmoid'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=5

finetune=1 # If you don't want to use fintuned model, set `finetune=0`.
model_choice="best"

batch_size=64
epochs_train=100
epochs_finetune=100

use_cuda=1
overwrite=0
seed_train=111
seed_finetune=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

prefix=""

if [ ${enc_basis} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_basis} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}"
    save_dir="${save_dir}/N${N}_L${L}_H${H}_X${X}_R${R}/${prefix}causal${causal}_mask-${mask_nonlinear}"
    if [ ${finetune} -eq 1 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs_train}+${epochs_finetune}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs_train}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}"
    fi
    save_dir="${save_dir}/seed${seed_train}"
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
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed_train} | tee "${log_dir}/test_${time_stamp}.log"
