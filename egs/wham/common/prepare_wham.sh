#!/bin/bash

dataset_root="../../../dataset"

. ./parse_options.sh || exit 1

file="wham_noise.zip"
wget "https://storage.googleapis.com/whisper-public/${file}" -P "/tmp"
unzip "/tmp/${file}" -d "${dataset_root}"
rm "/tmp/${file}"
