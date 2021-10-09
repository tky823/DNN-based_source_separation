#!/bin/bash

model_id=""
model_name="improved"
file="model"

. ./path.sh
. parse_options.sh || exit 1

echo "Download D3Net. (Dataset: MUSDB18, sampling frequency 44.1kHz)"

model_ids=(
    ["improved"]="1XL5RIoyIEnATH2zIO0BOHujWo7UnxwuY"
    ["paper"]="1UXtrJ0eo__Gonwqe9FoJaPQY4kFg_6aE"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${model_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${model_id}" -o "${file}.zip"

unzip "${file}.zip"