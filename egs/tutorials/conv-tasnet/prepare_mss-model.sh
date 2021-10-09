#!/bin/bash

model_id=""
model_name="sr44100_2sec"
file="model"

. ./path.sh
. parse_options.sh || exit 1

echo "Download Conv-TasNet. (Dataset: MUSDB18, sampling frequency 44.1kHz)"

model_ids=(
    ["sr44100_2sec"]="1kKQgkrG2FFnVOsuWeSBshFD1rhYT7lcq"
    ["sr44100_8sec"]="1yQTdGu2jigAHotJJ7JAQcfqedACCP40t"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"