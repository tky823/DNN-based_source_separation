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

train_wav_root="../../../dataset/WHAM/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tr"
valid_wav_root="../../../dataset/WHAM/${n_sources}speakers/wav${sr_k}k/${max_or_min}/cv"

train_list_path="../../../dataset/WHAM/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tr_mix"
valid_list_path="../../../dataset/WHAM/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_cv_mix"

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

task='separate-noisy'
prefix=""

if [ ${enc_basis} = 'trainable' -a -n "${enc_nonlinear}" ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${task}/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_H${H}_X${X}/${prefix}dropout${sep_dropout}_causal${causal}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
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

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train_separate-noisy.py \
--train_wav_root "${train_wav_root}" \
--valid_wav_root "${valid_wav_root}" \
--train_list_path "${train_list_path}" \
--valid_list_path "${valid_list_path}" \
--sample_rate ${sample_rate} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--enc_basis ${enc_basis} \
--dec_basis ${dec_basis} \
--enc_nonlinear "${enc_nonlinear}" \
-N ${N} \
-L ${L} \
-H ${H} \
-X ${X} \
--sep_dropout ${sep_dropout} \
--causal ${causal} \
--mask_nonlinear ${mask_nonlinear} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--max_norm ${max_norm} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/train_${time_stamp}.log"
