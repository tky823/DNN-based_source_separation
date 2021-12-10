#!/bin/bash

model_id=""
model_name="musdb18"

. ./path.sh
. parse_options.sh || exit 1

echo "Download CrossNet-Open-Unmix. (Dataset: MUSDB18, sampling frequency 44.1kHz)"

declare -A model_ids=(
    ["musdb18"]="1yQC00DFvHgs4U012Wzcg69lvRxw5K9Jj"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

# Download
python -c "from utils.utils import download_pretrained_model_from_google_drive as download; download('${model_id}', path='./')"