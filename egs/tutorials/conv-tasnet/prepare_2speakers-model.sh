#!/bin/bash

model_id="1dVTNcmmpdh-5IL8NGViVR6lE6QQhaKQo"

. ./path.sh
. parse_options.sh || exit 1

echo "Download Conv-TasNet. (Dataset: LibriSpeech, sampling frequency 16kHz)"

file='archive'

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"