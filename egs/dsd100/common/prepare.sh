#!/usr/bin/bash

DATA_DIR=$1
OUT_DIR=$2
samples=$3

data_file="DSD100.zip"
dataset_url="http://liutkus.net/${data_file}"

DATASET_ROOT="${DATA_DIR}/DSD100"
mixture_dataset="Mixtures"
source_dataset="Sources"

if [ -e "${DATASET_ROOT}" ]; then
    echo "Already downloaded dataset ${DATASET_ROOT}"
else
    mkdir -p "${DATA_DIR}"
    wget ${dataset_url} -P "./tmp"
    tar -xf "./tmp/${data_file}" -C "${DATA_DIR}"
    rm "./tmp/${data_file}"
fi

if [ ! -e "${OUT_DIR}/train/train.json" ]; then
    prepare_dsd100.py \
    --source_dir "${DATASET_ROOT}/${source_dataset}/Dev" \
    --mixture_dir "${DATASET_ROOT}/${mixture_dataset}/Dev" \
    --out_dir "${OUT_DIR}" \
    --samples ${samples} \
    --train_json_path "${OUT_DIR}/train/train.json"
fi

if [ ! -e "${OUT_DIR}/test/test.json" ]; then
    prepare_dsd100.py \
    --source_dir "${DATASET_ROOT}/${source_dataset}/Test" \
    --mixture_dir "${DATASET_ROOT}/${mixture_dataset}/Test" \
    --out_dir "${OUT_DIR}" \
    --test_json_path "${OUT_DIR}/test/test.json"
fi
