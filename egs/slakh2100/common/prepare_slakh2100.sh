#!/bin/bash

slakh2100_root="../../../dataset/babyslakh_16k"
is_tiny=0

. ./parse_options.sh || exit 1

mkdir -p "${slakh2100_root}"

if [ ${is_tiny} -eq 0 ]; then
    file="slakh2100_flac_redux.tar.gz"
    wget "https://zenodo.org/record/4599666/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "/tmp/"
    rm "/tmp/${file}"

    mv "/tmp/slakh2100_flac_redux/"* "${slakh2100_root}"
else
    file="babyslakh_16k.tar.gz"
    wget "https://zenodo.org/record/4603870/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "/tmp/"
    rm "/tmp/${file}"

    mv "/tmp/babyslakh_16k/"* "${slakh2100_root}"
fi
