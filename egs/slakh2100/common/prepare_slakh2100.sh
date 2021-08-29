#!/bin/bash

dataset_root="../../../dataset"
is_tiny=0

. ./parse_options.sh || exit 1

mkdir -p "${dataset_root}"

if [ ${is_tiny} -eq 0 ]; then
    file="slakh2100_flac_redux.tar.gz"
    wget "https://zenodo.org/record/4599666/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "${dataset_root}"
    rm "/tmp/${file}"
else
    file="babyslakh_16k.tar.gz"
    wget "https://zenodo.org/record/4603870/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "${dataset_root}"
    rm "/tmp/${file}"
fi
