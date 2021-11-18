#!/bin/bash

model_id=""
model_name="paper"

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

# Download
python -c "from utils.utils import download_pretrained_model_from_google_drive as download; download('${model_id}', path='./')"