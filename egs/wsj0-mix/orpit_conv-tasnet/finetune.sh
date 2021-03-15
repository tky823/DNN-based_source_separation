#!/bin/bash

exp_dir="./exp"
continue_from=""

n_sources='2+3'
n_sources_finetune=3
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=4
valid_duration=4
max_or_min='min'

train_wav_root="../../../dataset/wsj0-mix/${n_sources_finetune}speakers/wav${sr_k}k/${max_or_min}/tr"
valid_wav_root="../../../dataset/wsj0-mix/${n_sources_finetune}speakers/wav${sr_k}k/${max_or_min}/cv"
train_list_path="../../../dataset/wsj0-mix/${n_sources_finetune}speakers/mix_${n_sources_finetune}_spk_${max_or_min}_tr_mix"
valid_list_path="../../../dataset/wsj0-mix/${n_sources_finetune}speakers/mix_${n_sources_finetune}_spk_${max_or_min}_cv_mix"

# Encoder & decoder
enc_bases='trainable' # choose from 'trainable','Fourier', or 'trainableFourier'
dec_bases='trainable' # choose from 'trainable','Fourier', 'trainableFourier', or 'pinv'
enc_nonlinear='' # enc_nonlinear is activated if enc_bases='trainable' and dec_bases!='pinv'
window_fn='' # window_fn is activated if enc_bases='Fourier' or dec_bases='Fourier'
N=512
L=16

# Separator
H=512
B=128
Sc=128
P=3
X=8
R=3
dilated=1
separable=1
causal=0
sep_nonlinear='prelu'
sep_norm=1
mask_nonlinear='softmax'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=1e-5

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111

. ./path.sh
. parse_options.sh || exit 1

prefix=""

if [ ${enc_bases} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_bases} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ ${enc_bases} = 'Fourier' -o ${dec_bases} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_bases}-${dec_bases}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}/seed${seed}"

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="0"

finetune.py \
--train_wav_root ${train_wav_root} \
--valid_wav_root ${valid_wav_root} \
--train_list_path ${train_list_path} \
--valid_list_path ${valid_list_path} \
--sr ${sr} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--enc_bases ${enc_bases} \
--dec_bases ${dec_bases} \
--enc_nonlinear "${enc_nonlinear}" \
--window_fn "${window_fn}" \
-N ${N} \
-L ${L} \
-B ${B} \
-H ${H} \
-Sc ${Sc} \
-P ${P} \
-X ${X} \
-R ${R} \
--dilated ${dilated} \
--separable ${separable} \
--causal ${causal} \
--sep_nonlinear ${sep_nonlinear} \
--sep_norm ${sep_norm} \
--mask_nonlinear ${mask_nonlinear} \
--n_sources ${n_sources_finetune} \
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
