#!/bin/bash

. ./path.sh

DATA_DIR=$1
OUT_DIR=$2

data_file="DSD100.zip"
dataset_url="http://liutkus.net/${data_file}"

DATASET_ROOT="${DATA_DIR}/DSD100"

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
    --source_dir "${DATASET_ROOT}/Sources/Dev" \
    --mixture_dir "${DATASET_ROOT}/Mixtures/Dev" \
    --json_path "${OUT_DIR}/train/train.json"
fi

if [ ! -e "${OUT_DIR}/test/test.json" ]; then
    prepare_dsd100.py \
    --source_dir "${DATASET_ROOT}/Sources/Test" \
    --mixture_dir "${DATASET_ROOT}/Mixtures/Test" \
    --json_path "${OUT_DIR}/test/test.json"
fi
