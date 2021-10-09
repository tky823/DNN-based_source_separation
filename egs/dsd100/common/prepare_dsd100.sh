#!/bin/bash

dsd100_root="../../../dataset/DSD100"
json_dir="../../../dataset/DSD100"

. ./path.sh
. parse_options.sh

file="DSD100.zip"

if [ -e "${dsd100_root}/Sources/Test/001 - ANiMAL - Clinic A/bass.wav" ]; then
    echo "Already downloaded dataset ${dsd100_root}"
else
    if [ ! -d "${dsd100_root}" ] ; then
        mkdir -p "${dsd100_root}"
    fi
    wget "http://liutkus.net/${file}" -P "./tmp"
    tar -xzvf "/tmp/${file}" -C "/tmp/"
    mv "/tmp/DSD100/"* "${dsd100_root}/"
    rm "/tmp/${file}"
fi

if [ ! -e "${json_dir}/train/train.json" ]; then
    prepare_dsd100.py \
    --source_dir "${dsd100_root}/Sources/Dev" \
    --mixture_dir "${dsd100_root}/Mixtures/Dev" \
    --json_path "${json_dir}/train/train.json"
fi

if [ ! -e "${json_dir}/test/test.json" ]; then
    prepare_dsd100.py \
    --source_dir "${dsd100_root}/Sources/Test" \
    --mixture_dir "${dsd100_root}/Mixtures/Test" \
    --json_path "${json_dir}/test/test.json"
fi
