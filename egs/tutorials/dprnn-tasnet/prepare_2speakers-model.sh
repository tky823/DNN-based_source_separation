#!/bin/bash

model_id=""
model_name="sr16000_L16_librispeech"
file="model"

. ./path.sh
. parse_options.sh || exit 1

echo "Download DPRNN-TasNet. (Dataset: LibriSpeech, sampling frequency 16kHz)"

declare -A model_ids=(
    ["sr16000_L16_librispeech"]="1hTmxhI8JQlNnWVjwWUBGYlC7O_-ykK4H"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"