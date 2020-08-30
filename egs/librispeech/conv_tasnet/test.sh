#!/bin/bash

. ./path.sh

exp_dir="$1"

n_sources=2

wav_root="../../../dataset/LibriSpeech"
test_json_path="../../../dataset/LibriSpeech/test-clean/test-${n_sources}mix.json"

sr=16000

# Encoder & decoder
enc_basis='trainable'
dec_basis='trainable'
enc_nonlinear='relu' # window_fn is activated if enc_basis='trainable'
window_fn='hamming' # window_fn is activated if enc_basis='Fourier' or dec_basis='Fourier'
N=64
L=16

# Separator
H=256
B=128
Sc=128
P=3
X=6
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
lr=0.001
weight_decay=0
max_norm=5

batch_size=4
epochs=100

use_cuda=0
overwrite=0
seed=111

prefix=""

if [ ${enc_basis} = 'trainable' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ ${enc_basis} = 'Fourier' -o ${dec_basis} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

save_dir="${exp_dir}/${n_sources}mix/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_choice="best"

model_dir="${save_dir}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/log"
out_dir="${save_dir}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

test.py \
--wav_root ${wav_root} \
--test_json_path ${test_json_path} \
--sr ${sr} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
