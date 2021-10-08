#!/bin/bash

model_id="1yQTdGu2jigAHotJJ7JAQcfqedACCP40t"

. ./path.sh
. parse_options.sh || exit 1

echo "Download Conv-TasNet. (Dataset: LibriSpeech, sampling frequency 44.1kHz)"

file='archive'

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"