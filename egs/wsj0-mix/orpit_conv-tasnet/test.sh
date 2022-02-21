#!/bin/bash

exp_dir="./exp"
tag=""

n_sources_train='2+3'
n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sample_rate=${sr_k}000
duration=4
max_or_min='min'

test_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

# Encoder & decoder
enc_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
dec_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase', 'pinv']
enc_nonlinear='' # enc_nonlinear is activated if enc_basis='trainable' and dec_basis!='pinv'
window_fn='' # window_fn is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_onesided=0 # enc_onesided is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_return_complex=0 # enc_return_complex is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']

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
max_norm=5

finetune=1 # If you don't want to use fintuned model, set `finetune=0`.
model_choice="best"

batch_size_train=4
batch_size_finetune=4
epochs_train=100
epochs_finetune=10

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

if [ ${enc_basis} = 'Fourier' -o ${enc_basis} = 'trainableFourier' -o ${enc_basis} = 'trainableFourierTrainablePhase' -o ${dec_basis} = 'Fourier' -o ${dec_basis} = 'trainableFourier' -o ${dec_basis} = 'trainableFourierTrainablePhase' ]; then
    prefix="${preffix}${window_fn}-window_enc-onesided${enc_onesided}_enc-complex${enc_return_complex}/"
fi

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${n_sources_train}mix/sr${sr_k}k_${max_or_min}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size_train}_e${epochs_train}_${optimizer}-lr${lr}-decay${weight_decay}/seed${seed_train}"
else
    save_dir="${exp_dir}/${tag}"
fi

if [ ${finetune} -eq 1 ]; then
    save_dir="${save_dir}/finetune/b${batch_size_finetune}_e${epochs_finetune}/seed${seed_finetune}"
fi

model_dir="${save_dir}/model"
model_path="${model_dir}/${model_choice}.pth"
log_dir="${save_dir}/log"
out_dir="${save_dir}/test/${n_sources}mix"

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
