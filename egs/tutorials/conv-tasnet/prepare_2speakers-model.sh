#!/bin/bash

model_id=""
model_name="sr16000_L16_librispeech"

. ./path.sh
. parse_options.sh || exit 1

echo "Download Conv-TasNet. (Dataset: LibriSpeech, sampling frequency 16kHz)"

declare -A model_ids=(
    ["sr16000_L16_librispeech"]="1NI6Q_WZHiTKkgkNTEcZE1yHskHgYUHpy"
)

if [ -z "${model_id}" ] ; then
    model_id="${model_ids[${model_name}]}"
fi

# Download
python -c "from utils.utils import download_pretrained_model_from_google_drive as download; download('${model_id}', path='./')"