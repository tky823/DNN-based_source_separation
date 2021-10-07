#!/bin/bash

wham_root="../../../dataset/WHAM"
wham_2speakers_root="../../../dataset/WHAM/2speakers"
wham_noise_root="../../../dataset/wham_noise"
create_from="wsjmix" # or "scratch"

# WSJ0
wsj0_root="../../../dataset/wsj0_wav"
wsjmix_2speakers_8k="../../../dataset/wsj0-mix/2speakers/wav8k"
wsjmix_2speakers_16k="../../../dataset/wsj0-mix/2speakers/wav16k"

. ./parse_options.sh || exit 1

# Prepare wham noise
. ./prepare_wham_noise.sh \
--wham_noise_root "${wham_noise_root}"

# Prepare wham mixture
file="wham_scripts.tar.gz"

if [ -e "${wham_root}/wham_scripts/create_wham_from_scratch.py" ] ; then
    echo "Already downloaded wham_scripts."
else
    if [ ! -d "${wham_root}" ] ; then
        mkdir -p "${wham_root}"
    fi
    wget "https://storage.googleapis.com/whisper-public/${file}" -P "/tmp/"
    tar -xzvf "/tmp/${file}" -C "${wham_root}"
    rm "/tmp/${file}"
fi

work_dir="$PWD"

cd "${wham_root}/wham_scripts/"

if [ "${create_from}" = "scratch" ] ; then
    python create_wham_from_scratch.py \
    --wsj0-root "${wsj0_root}" \
    --wham-noise-root "${wham_noise_root}" \
    --output-dir "${wham_2speakers_root}"
elif [ "${create_from}" = "wsjmix" ] ; then
    python create_wham_from_wsjmix.py \
    --wsjmix-dir-8k "${wsjmix_2speakers_8k}" \
    --wsjmix-dir-16k "${wsjmix_2speakers_16k}" \
    --wham-noise-root "${wham_noise_root}" \
    --output-dir "${wham_2speakers_root}"
else
    echo "'create_from' is expected scratch or wsjmix."
fi

cd "${work_dir}"