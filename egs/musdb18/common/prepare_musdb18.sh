#!/bin/bash

musdb18_root="../../../dataset/musdb18"
musdb18hq_root="../../../dataset/musdb18hq"
file=musdb18.zip
is_hq=0

. ./parse_options.sh || exit 1

if [ ${is_hq} -eq 0 ]; then
    if [ -e "${musdb18_root}/train/A Classic Education - NightOwl.stem.mp4" ]; then
        echo "Already downloaded dataset ${musdb18_root}"
    else
        mkdir -p "${musdb18_root}"
        wget "https://zenodo.org/record/1117372/files/${file}" -P "/tmp"
        unzip "${file}" -d "${musdb18_root}"
        rm "/tmp/${file}"
    fi
else
    if [ -e "${musdb18hq_root}/train/A Classic Education - NightOwl.stem.mp4" ]; then
        echo "Already downloaded dataset ${musdb18hq_root}"
    else
        mkdir -p "${musdb18hq_root}"
        wget "https://zenodo.org/record/3338373/files/${file}" -P "/tmp"
        unzip "${file}" -d "${musdb18hq_root}"
        rm "/tmp/${file}"
    fi
fi