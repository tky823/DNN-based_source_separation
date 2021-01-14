#!/bin/bash

. ./path.sh

exp_dir="$1"

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=2
valid_duration=4
max_or_min='min'

wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

# Encoder & decoder
enc_bases='trainable' # choose from 'trainable','Fourier', or 'trainableFourier'
dec_bases='trainable' # choose from 'trainable','Fourier', 'trainableFourier', or 'pinv'
enc_nonlinear='relu' # enc_nonlinear is activated if enc_bases='trainable' and dec_bases!='pinv'
window_fn='hamming' # window_fn is activated if enc_bases='Fourier' or dec_bases='Fourier'
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

use_cuda=0
overwrite=0
seed=111

prefix=""

if [ ${enc_bases} = 'trainable' -a ${dec_bases} -ne 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

if [ ${enc_bases} = 'Fourier' -o ${dec_bases} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

save_dir="${exp_dir}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_bases}-${dec_bases}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_choice="best"

model_dir="${save_dir}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/log"
out_dir="${save_dir}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="0"

test.py \
--wav_root ${wav_root} \
--test_list_path ${test_list_path} \
--sr ${sr} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
