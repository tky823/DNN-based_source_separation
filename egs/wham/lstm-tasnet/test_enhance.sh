#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2
n_target_speakers=1
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sample_rate=${sr_k}000
duration=4
max_or_min='min'

test_wav_root="../../../dataset/WHAM/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/WHAM/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

# Encoder & decoder
enc_basis='trainable' # choose from ['trainable', 'trainableGated']
dec_basis='trainable' # choose from ['trainable']
enc_nonlinear='' # enc_nonlinear is activated if enc_basis='trainable'

N=500
L=80

# Separator
H=600
X=4
sep_dropout=0.3
causal=0
mask_nonlinear='softmax'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=3e-4
weight_decay=0
max_norm=5

batch_size=128
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

task='enhance'

if [ ${n_target_speakers} -eq 1 ] ; then
    task_detail='enhance-single'
elif [ ${n_target_speakers} -eq 2 ] ; then
    task_detail='enhance-both'
else
    echo "n_target_speakers is expected 1 or 2 but given ${n_target_speakers}."
    exit 1
fi

prefix=""

if [ ${enc_basis} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_basis} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${task}/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_H${H}_X${X}/${prefix}dropout${sep_dropout}_causal${causal}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
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

test_enhance.py \
--test_wav_root "${test_wav_root}" \
--test_list_path "${test_list_path}" \
--sample_rate ${sample_rate} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
