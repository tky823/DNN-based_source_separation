#!/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

sources="[bass,drums,other,vocals]"
duration=8
valid_duration=100

musdb18_root="../../../dataset/MUSDB18"
sample_rate=44100

# Encoder & decoder
enc_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
dec_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase', 'pinv']
enc_nonlinear='' # enc_nonlinear is activated if enc_basis='trainable' and dec_basis!='pinv'
window_fn='' # window_fn is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_onesided=0 # enc_onesided is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_return_complex=0 # enc_return_complex is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']

N=256
L=20

# Separator
H=512
B=256
Sc=128
P=3
X=10
R=4
dilated=1
separable=1
causal=0
sep_nonlinear='prelu'
sep_norm=1
mask_nonlinear='sigmoid'

# Augmentation
augmentation_path="./config/paper/augmentation.yaml"

# Criterion
criterion='mse'

# Optimizer
optimizer='adam'
lr=3e-4
weight_decay=0
max_norm=5

batch_size=4
samples_per_epoch=-1
epochs=100

estimate_all=1
evaluate_all=1

use_cuda=1
seed=111
gpu_id="0"

model_choice="best" # 'last' or 'best'

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
    save_dir="${exp_dir}/${sources}/sr${sample_rate}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}"
    if [ ${samples_per_epoch} -gt 0 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs}-s${samples_per_epoch}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    fi
else
    save_dir="${exp_dir}/${tag}"
fi

model_path="${save_dir}/model/${model_choice}.pth"
log_dir="${save_dir}/log/test/${model_choice}"
json_dir="${save_dir}/json/${model_choice}"

musdb=`basename "${musdb18_root}"` # 'MUSDB18' or 'MUSDB18HQ'
estimates_dir="${save_dir}/${musdb}/${model_choice}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--musdb18_root ${musdb18_root} \
--sample_rate ${sample_rate} \
--duration ${duration} \
--sources ${sources} \
--criterion ${criterion} \
--estimates_dir "${estimates_dir}" \
--json_dir "${json_dir}" \
--model_path "${model_path}" \
--estimate_all ${estimate_all} \
--evaluate_all ${evaluate_all} \
--use_cuda ${use_cuda} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"
