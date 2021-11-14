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

use_cuda=1
overwrite=0
num_workers=2
seed=111
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
    save_dir="${exp_dir}/${sources}/sr${sample_rate}/${duration}sec/${enc_basis}-${dec_basis}/${criterion}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_norm${sep_norm}_mask-${mask_nonlinear}"
    if [ ${samples_per_epoch} -gt 0 ]; then
        save_dir="${save_dir}/b${batch_size}_e${epochs}-s${samples_per_epoch}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    else
        save_dir="${save_dir}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
    fi
else
    save_dir="${exp_dir}/${tag}"
fi

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
config_dir="${save_dir}/config"
log_dir="${save_dir}/log"

if [ ! -e "${config_dir}" ]; then
    mkdir -p "${config_dir}"
fi

augmentation_dir=`dirname ${augmentation_path}`
augmentation_name=`basename ${augmentation_path}`

if [ ! -e "${config_dir}/${augmentation_name}" ]; then
    cp "${augmentation_path}" "${config_dir}/${augmentation_name}"
fi

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train.py \
--musdb18_root ${musdb18_root} \
--sample_rate ${sample_rate} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--augmentation_path "${augmentation_path}" \
--enc_basis ${enc_basis} \
--dec_basis ${dec_basis} \
--enc_nonlinear "${enc_nonlinear}" \
--window_fn "${window_fn}" \
--enc_onesided "${enc_onesided}" \
--enc_return_complex "${enc_return_complex}" \
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
--sources ${sources} \
--criterion ${criterion} \
--optimizer ${optimizer} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--max_norm ${max_norm} \
--batch_size ${batch_size} \
--samples_per_epoch ${samples_per_epoch} \
--epochs ${epochs} \
--model_dir "${model_dir}" \
--loss_dir "${loss_dir}" \
--sample_dir "${sample_dir}" \
--continue_from "${continue_from}" \
--use_cuda ${use_cuda} \
--overwrite ${overwrite} \
--num_workers ${num_workers} \
--seed ${seed} | tee "${log_dir}/train_${time_stamp}.log"
