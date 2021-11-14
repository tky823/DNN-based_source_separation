#!/bin/bash

model_id=""
model_name="musdb18"
file="model"

. ./path.sh
. parse_options.sh || exit 1

echo "Download Open-Unmix. (Dataset: MUSDB18, sampling frequency 44.1kHz)"

declare -A model_ids=(
    ["musdb18"]="1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T"
    ["musdb18hq"]="18pj2ubYnZPSQWPpHaREAcbmrNzEihNHO"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"