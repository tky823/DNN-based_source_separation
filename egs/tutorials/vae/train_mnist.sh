#!/usr/bin/bash

exp_dir="./exp"
continue_from=""
tag=""

dataset_root="../../../dataset"
config_path="./config/baseline_mnist.yaml"

latent_dim=10
L=1 # Number of samples
H=200
R=3

optimizer="adam"
lr=1e-3
weight_decay=0
max_norm=0

batch_size=100
epochs=100

use_cuda=1
overwrite=0
seed=42

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/latent${latent_dim}_L${L}_H${H}_R${R}/b${batch_size}_e${epochs}_${optimizer}-lr${lr}-decay${weight_decay}_clip${max_norm}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

model_dir="${save_dir}/model"
loss_dir="${save_dir}/loss"
sample_dir="${save_dir}/sample"

train_mnist.py \
--dataset_root "${dataset_root}" \
--config_path "${config_path}" \
--latent_dim ${latent_dim} \
--hidden_channels ${H} \
--num_layers ${R} \
--num_samples ${L} \
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