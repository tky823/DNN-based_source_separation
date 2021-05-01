#!/bin/bash

dataset_root="../../../dataset"
n_sources=2

sr=16000
train_duration=2
valid_duration=4
test_duration=4

seed=111

. ./path.sh
. ./parse_options.sh || exit 1

train_file="train-clean-100.tar.gz"
train_url="http://www.openslr.org/resources/12/${train_file}"
train_dataset="train-clean-100"
train_json="train-100-${n_sources}mix.json"
valid_file="dev-clean.tar.gz"
valid_url="http://www.openslr.org/resources/12/${valid_file}"
valid_dataset="dev-clean"
valid_json="valid-${n_sources}mix.json"
test_file="test-clean.tar.gz"
test_url="http://www.openslr.org/resources/12/${test_file}"
test_dataset="test-clean"
test_json="test-${n_sources}mix.json"

if [ -e "${dataset_root}/LibriSpeech/${train_dataset}/103" ]; then
    echo "Already downloaded dataset ${train_dataset}"
else
    mkdir -p "${dataset_root}"
    wget ${train_url} -P "/tmp"
    tar -xf "/tmp/${train_file}" -C "${dataset_root}"
    rm "/tmp/${train_file}"
fi

if [ -e "${dataset_root}/LibriSpeech/${valid_dataset}/1272" ]; then
    echo "Already downloaded dataset ${valid_dataset}"
else
    mkdir -p "${dataset_root}"
    wget ${valid_url} -P "/tmp"
    tar -xf "/tmp/${valid_file}" -C "${dataset_root}"
    rm "/tmp/${valid_file}"
fi

if [ -e "${dataset_root}/LibriSpeech/${test_dataset}/1089" ]; then
    echo "Already downloaded dataset ${test_dataset}"
else
    mkdir -p "${dataset_root}"
    wget ${test_url} -P "/tmp"
    tar -xf "/tmp/${test_file}" -C "${dataset_root}"
    rm "/tmp/${test_file}"
fi

if [ -e "${dataset_root}/LibriSpeech/${train_dataset}/${train_json}" ]; then
    echo "${train_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${dataset_root}/LibriSpeech" \
    --wav_root "${dataset_root}/LibriSpeech" \
    --json_path "${dataset_root}/LibriSpeech/${train_dataset}/${train_json}" \
    --n_sources ${n_sources} \
    --sr ${sr} \
    --duration ${train_duration} \
    --seed ${seed}
fi

if [ -e "${dataset_root}/LibriSpeech/${valid_dataset}/${valid_json}" ]; then
    echo "${valid_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${dataset_root}/LibriSpeech" \
    --wav_root "${dataset_root}/LibriSpeech" \
    --json_path "${dataset_root}/LibriSpeech/${valid_dataset}/${valid_json}" \
    --n_sources ${n_sources} \
    --sr ${sr} \
    --duration ${valid_duration} \
    --seed ${seed}
fi

if [ -e "${dataset_root}/LibriSpeech/${test_dataset}/${test_json}" ]; then
    echo "${test_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${dataset_root}/LibriSpeech" \
    --wav_root "${dataset_root}/LibriSpeech" \
    --json_path "${dataset_root}/LibriSpeech/${test_dataset}/${test_json}" \
    --n_sources ${n_sources} \
    --sr ${sr} \
    --duration ${test_duration} \
    --seed ${seed}
fi
