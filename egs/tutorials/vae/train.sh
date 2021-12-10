#!/usr/bin/bash

exp_dir="./exp"
tag=""

latent_dim=10
L=1 # Number of samples
H=200
R=3

batch_size=100
lr=1e-3
epochs=100

. ./path.sh

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/latent${latent_dim}_L${L}_H${H}_R${R}_b${batch_size}_lr${lr}_epochs${epochs}/seed${seed}"
else
    save_dir="${exp_dir}/${tag}"
fi

train.py \
--latent_dim ${latent_dim} \
--hidden_channels ${H} \
--num_layers ${R} \
--num_samples ${L} \
--batch_size ${batch_size} \
--lr ${lr} \
--epochs ${epochs} \
--save_dir "${exp_dir}"