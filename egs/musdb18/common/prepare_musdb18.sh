#!/bin/bash

musdb18_root="../../../dataset/musdb18"
file=musdb18.zip

. ./parse_options.sh || exit 1

if [ -e "${musdb18_root}/train/A Classic Education - NightOwl.stem.mp4" ]; then
    echo "Already downloaded dataset ${musdb18_root}"
else
    mkdir -p "${musdb18_root}/train"
    wget "https://zenodo.org/record/1117372/files/${file}" -P "/tmp"
    # wget "https://zenodo.org/record/3338373/files/${file}" -P "/tmp"
    unzip "${file}" -d "${musdb18_root}"
    rm "/tmp/${file}"
fi


