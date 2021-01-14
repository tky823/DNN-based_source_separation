#!/bin/bash

. ./path.sh

exp_dir="$1"

n_sources=2

sr=16000

# Encoder & decoder
enc_bases='trainable' # choose from 'trainable','Fourier', or 'trainableFourier'
dec_bases='trainable' # choose from 'trainable','Fourier', 'trainableFourier', or 'pinv'
enc_nonlinear='relu' # enc_nonlinear is activated if enc_bases='trainable' and dec_bases!='pinv'
window_fn='hamming' # window_fn is activated if enc_bases='Fourier' or dec_bases='Fourier'
N=64
L=16

# Separator
H=256
K=100
P=50
B=3
dilated=1
separable=1
causal=0
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

if [ -z "${exp_dir}" ]; then
    exp_dir="./exp"
fi

prefix=""

if [ ${enc_bases} = 'trainable' -a ${dec_bases} -ne 'pinv']; then
    prefix="${preffix}enc-${enc_nonlinear}_"    
fi

if [ ${enc_bases} = 'Fourier' -o ${dec_bases} = 'Fourier' ]; then
    prefix="${preffix}${window_fn}-window_"
fi

save_dir="${exp_dir}/${n_sources}mix/${enc_bases}-${dec_bases}/${criterion}/N${N}_L${L}_H${H}_K${K}_P${P}_B${B}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_choice="best"
model_path="${save_dir}/model/${model_choice}.pth"

num_chunk=256
duration=5

demo.py \
--sr ${sr} \
--num_chunk ${num_chunk} \
--duration ${duration} \
--model_path "${model_path}" \
--save_dir "./results"
