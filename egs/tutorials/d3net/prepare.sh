#!/bin/bash

model_id=""
model_name="paper"
file="model"

. ./path.sh
. parse_options.sh || exit 1

echo "Download D3Net. (Dataset: MUSDB18, sampling frequency 44.1kHz)"

declare -A model_ids=(
    ["paper"]="1We9ea5qe3Hhcw28w1XZl2KKogW9wdzKF"
    ["nnabla"]="1B4e4e-8-T1oKzSg8WJ8RIbZ99QASamPB"
    ["improved"]="1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"