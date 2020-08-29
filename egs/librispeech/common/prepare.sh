#!/bin/bash

. ./path.sh

DATASET_ROOT=$1
N_SOURCES=$2

sr=16000
train_duration=2
valid_duration=2
test_duration=2

seed=111

train_file="train-clean-100.tar.gz"
train_url="http://www.openslr.org/resources/12/${train_file}"
train_dataset="train-clean-100"
train_json="train-100-${N_SOURCES}mix.json"
valid_file="dev-clean.tar.gz"
valid_url="http://www.openslr.org/resources/12/${valid_file}"
valid_dataset="dev-clean"
valid_json="valid-${N_SOURCES}mix.json"
test_file="test-clean.tar.gz"
test_url="http://www.openslr.org/resources/12/${test_file}"
test_dataset="test-clean"
test_json="test-${N_SOURCES}mix.json"

if [ -e "${DATASET_ROOT}/LibriSpeech/${train_dataset}" ]; then
    echo "Already downloaded dataset ${train_dataset}"
else
    mkdir -p "${DATASET_ROOT}"
    wget ${train_url} -P "/tmp"
    tar -xf "/tmp/${train_file}" -C "${DATASET_ROOT}"
    rm "/tmp/${train_file}"
fi

if [ -e "${DATASET_ROOT}/LibriSpeech/${valid_dataset}" ]; then
    echo "Already downloaded dataset ${valid_dataset}"
else
    mkdir -p "${DATASET_ROOT}"
    wget ${valid_url} -P "/tmp"
    tar -xf "/tmp/${valid_file}" -C "${DATASET_ROOT}"
    rm "/tmp/${valid_file}"
fi

if [ -e "${DATASET_ROOT}/LibriSpeech/${test_dataset}" ]; then
    echo "Already downloaded dataset ${test_dataset}"
else
    mkdir -p "${DATASET_ROOT}"
    wget ${test_url} -P "/tmp"
    tar -xf "/tmp/${test_file}" -C "${DATASET_ROOT}"
    rm "/tmp/${test_file}"
fi

if [ -e "${DATASET_ROOT}/LibriSpeech/${train_dataset}/${train_json}" ]; then
    echo "${train_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${DATASET_ROOT}/LibriSpeech" \
    --wav_root "${DATASET_ROOT}/LibriSpeech" \
    --json_path "${DATASET_ROOT}/LibriSpeech/${train_dataset}/${train_json}" \
    --n_sources ${N_SOURCES} \
    --sr ${sr} \
    --duration ${train_duration} \
    --seed ${seed}
fi

if [ -e "${DATASET_ROOT}/LibriSpeech/${valid_dataset}/${valid_json}" ]; then
    echo "${valid_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${DATASET_ROOT}/LibriSpeech" \
    --wav_root "${DATASET_ROOT}/LibriSpeech" \
    --json_path "${DATASET_ROOT}/LibriSpeech/${valid_dataset}/${valid_json}" \
    --n_sources ${N_SOURCES} \
    --sr ${sr} \
    --duration ${valid_duration} \
    --seed ${seed}
fi

if [ -e "${DATASET_ROOT}/LibriSpeech/${test_dataset}/${test_json}" ]; then
    echo "${test_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${DATASET_ROOT}/LibriSpeech" \
    --wav_root "${DATASET_ROOT}/LibriSpeech" \
    --json_path "${DATASET_ROOT}/LibriSpeech/${test_dataset}/${test_json}" \
    --n_sources ${N_SOURCES} \
    --sr ${sr} \
    --duration ${test_duration} \
    --seed ${seed}
fi
