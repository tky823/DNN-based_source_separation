#!/bin/bash

exp_dir="./exp"
continue_from=""

sources="[drums,bass,other,vocals]"
target='bass'
duration=4
valid_duration=10

musdb18_root="../../../dataset/musdb18"
sr=44100

# Encoder & decoder
enc_bases='trainable' # choose from 'trainable','Fourier', or 'trainableFourier'
dec_bases='trainable' # choose from 'trainable','Fourier', 'trainableFourier', or 'pinv'
enc_nonlinear='' # enc_nonlinear is activated if enc_bases='trainable' and dec_bases!='pinv'
window_fn='' # window_fn is activated if enc_bases='Fourier' or dec_bases='Fourier'
N=512
L=64

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
mask_nonlinear='sigmoid'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=5

batch_size=4
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

train_json_path="../../../dataset/musdb18/train/${target}/sr${sr}_${duration}sec.json"

prefix=""

if [ ${enc_bases} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_bases} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ ${enc_bases} = 'Fourier' -o ${dec_bases} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

save_dir="${exp_dir}/${target}/sr${sr}/${duration}sec/${enc_bases}-${dec_bases}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train_single.py \
--musdb18_root ${musdb18_root} \
--train_json_path "${train_json_path}" \
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
--target ${target} \
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
