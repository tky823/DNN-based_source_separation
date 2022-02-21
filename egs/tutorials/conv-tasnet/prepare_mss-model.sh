#!/bin/bash

model_id=""
model_name="sr44100_L20"

. ./path.sh
. parse_options.sh || exit 1

echo "Download Conv-TasNet. (Dataset: MUSDB18, sampling frequency 44.1kHz)"

declare -A model_ids=(
    ["sr44100_L20"]="1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ"
    ["sr44100_L64"]="1paXNGgH8m0kiJTQnn1WH-jEIurCKXwtw"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

# Download
python -c "from utils.utils import download_pretrained_model_from_google_drive as download; download('${model_id}', path='./')"