#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2

wav_root="../../../dataset/LibriSpeech"
test_json_path="../../../dataset/LibriSpeech/test-clean/test-${n_sources}mix.json"

sample_rate=16000

# Encoder & decoder
enc_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
dec_basis='trainable' # choose from ['trainable','Fourier', 'trainableFourier', 'trainableFourierTrainablePhase', 'pinv']
enc_nonlinear='relu' # enc_nonlinear is activated if enc_basis='trainable' and dec_basis!='pinv'
window_fn='' # window_fn is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_onesided=0 # enc_onesided is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']
enc_return_complex=0 # enc_return_complex is activated if enc_basis or dec_basis in ['Fourier', 'trainableFourier', 'trainableFourierTrainablePhase']

N=64
L=16 # L corresponds to the window length (samples) in this script.

# Separator
F=64
H=128
K=100
P=50
B=3
causal=0
sep_norm=1
mask_nonlinear='sigmoid'

# Criterion
criterion='sisdr'

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=0

batch_size=4
epochs=100

model_choice="best"

use_cuda=0
overwrite=0
seed=111

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
    save_dir="${exp_dir}/${n_sources}mix/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_F${F}_H${H}_K${K}_P${P}_B${B}/${prefix}causal${causal}_norm${sep_norm}_mask-${mask_nonlinear}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
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

export CUDA_VISIBLE_DEVICES="0"

test.py \
--wav_root ${wav_root} \
--test_json_path ${test_json_path} \
--sample_rate ${sample_rate} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--model_path "${model_path}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--seed ${seed} | tee "${log_dir}/test_${time_stamp}.log"

