#!/bin/bash

exp_dir="./exp"
continue_from=""

sources="[bass,drums,other,vocals]"
duration=8
valid_duration=10

musdb18_root="../../../dataset/musdb18"
is_wav=0 # If MUSDB is used, select 0. If MUSDB-HQ is used select 1.
sample_rate='[8000,16000,32000]'
stage=1

# Encoder & decoder
enc_bases='trainable' # choose 'trainable' only
dec_bases='trainable' # choose 'trainable' only
enc_nonlinear='' # enc_nonlinear is activated if enc_bases='trainable' and dec_bases!='pinv'
window_fn='hann'
N=440
L=20

# Separator
H=160
B=160
Sc=160
P=3
X=8
R=3
dilated=1
separable=1
causal=0
sep_nonlinear='prelu'
mask_nonlinear='sigmoid'

conv_name='generated'
norm_name='generated'

embed_dim=8
embed_bottleneck_channels=5
n_fft=1024
hop_length=256
enc_compression_rate=4
num_filters=6
n_mels=256
dropout=0

# Criterion
criterion_reconstruction='sisdr'
criterion_similarity='cos'
reconstruction=5e-2
similarity=2e+0
dissimilarity=3e+0

# Optimizer
optimizer='adam'
lr=1e-3
weight_decay=0
max_norm=5

batch_size=12
epochs=100

use_cuda=1
overwrite=0
seed=111
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

prefix=""

if [ ${enc_bases} = 'trainable' -a -n "${enc_nonlinear}" -a ${dec_bases} != 'pinv' ]; then
    prefix="${preffix}enc-${enc_nonlinear}_"
fi

save_dir="${exp_dir}/${sources}/sr${sample_rate}/${duration}sec/${enc_bases}-${dec_bases}/${criterion_reconstruction}-${reconstruction}_${criterion_similarity}${similarity}-${dissimilarity}/N${N}_L${L}_B${B}_H${H}_Sc${Sc}_P${P}_X${X}_R${R}/${prefix}dilated${dilated}_separable${separable}_causal${causal}_${sep_nonlinear}_mask-${mask_nonlinear}_conv-${conv_name}_norm-${norm_name}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"
log_dir="${save_dir}/log"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

train.py \
--musdb18_root "${musdb18_root}" \
--is_wav ${is_wav} \
--sample_rate ${sample_rate} \
--duration ${duration} \
--valid_duration ${valid_duration} \
--stage ${stage} \
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
--sep_nonlinear "${sep_nonlinear}" \
--mask_nonlinear "${mask_nonlinear}" \
--conv_name ${conv_name} \
--norm_name ${norm_name} \
--embed_dim ${embed_dim} \
--embed_bottleneck_channels ${embed_bottleneck_channels} \
--n_fft ${n_fft} \
--hop_length ${hop_length} \
--enc_compression_rate ${enc_compression_rate} \
--num_filters ${num_filters} \
--n_mels ${n_mels} \
--dropout ${dropout} \
--sources ${sources} \
--criterion_reconstruction ${criterion_reconstruction} \
--criterion_similarity ${criterion_similarity} \
--reconstruction ${reconstruction} \
--similarity ${similarity} \
--dissimilarity ${dissimilarity} \
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
--seed ${seed}