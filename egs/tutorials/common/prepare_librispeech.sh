#!/bin/bash

librispeech_root="../../../dataset"
n_sources=2

sample_rate=16000
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

if [ -e "${librispeech_root}/${train_dataset}/103" ]; then
    echo "Already downloaded dataset ${train_dataset}"
else
    if [ ! -d "${librispeech_root}/${train_dataset}" ] ; then
        mkdir -p "${librispeech_root}/${train_dataset}"
    fi
    wget ${train_url} -P "/tmp"
    tar -xf "/tmp/${train_file}" -C "/tmp/"
    rm "/tmp/${train_file}"
    mv "/tmp/LibriSpeech/${train_dataset}/"* "${librispeech_root}/${train_dataset}/"
fi

if [ -e "${librispeech_root}/${valid_dataset}/1272" ]; then
    echo "Already downloaded dataset ${valid_dataset}"
else
    if [ ! -d "${librispeech_root}/${valid_dataset}" ] ; then
        mkdir -p "${librispeech_root}/${valid_dataset}"
    fi
    wget ${valid_url} -P "/tmp"
    tar -xf "/tmp/${valid_file}" -C "/tmp/"
    rm "/tmp/${valid_file}"
    mv "/tmp/LibriSpeech/${valid_dataset}/"* "${librispeech_root}/${valid_dataset}/"
fi

if [ -e "${librispeech_root}/${test_dataset}/1089" ]; then
    echo "Already downloaded dataset ${test_dataset}"
else
    if [ ! -d "${librispeech_root}/${test_dataset}" ] ; then
        mkdir -p "${librispeech_root}/${test_dataset}"
    fi
    wget ${test_url} -P "/tmp"
    tar -xf "/tmp/${test_file}" -C "/tmp/"
    rm "/tmp/${test_file}"
    mv "/tmp/LibriSpeech/${test_dataset}/"* "${librispeech_root}/${test_dataset}/"
fi

if [ -e "${librispeech_root}/${train_dataset}/${train_json}" ]; then
    echo "${train_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${librispeech_root}" \
    --wav_root "${librispeech_root}" \
    --json_path "${librispeech_root}/${train_dataset}/${train_json}" \
    --n_sources ${n_sources} \
    --sample_rate ${sample_rate} \
    --duration ${train_duration} \
    --seed ${seed}
fi

if [ -e "${librispeech_root}/${valid_dataset}/${valid_json}" ]; then
    echo "${valid_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${librispeech_root}" \
    --wav_root "${librispeech_root}" \
    --json_path "${librispeech_root}/${valid_dataset}/${valid_json}" \
    --n_sources ${n_sources} \
    --sample_rate ${sample_rate} \
    --duration ${valid_duration} \
    --seed ${seed}
fi

if [ -e "${librispeech_root}/${test_dataset}/${test_json}" ]; then
    echo "${test_json} already exists."
else
    prepare_librispeech.py \
    --librispeech_root "${librispeech_root}" \
    --wav_root "${librispeech_root}" \
    --json_path "${librispeech_root}/${test_dataset}/${test_json}" \
    --n_sources ${n_sources} \
    --sample_rate ${sample_rate} \
    --duration ${test_duration} \
    --seed ${seed}
fi
