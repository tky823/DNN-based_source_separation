#!/bin/bash

wham_root="../../../dataset/WHAM!"
wham_noise_root="../../../dataset/wham_noise"
create_from="wsjmix" # or "scratch"

# WSJ0
wsj0_root="../../../dataset/wsj0_wav"
wsjmix_8k="../../../dataset/wsj0-mix/2speakers/wav8k"
wsjmix_16k="../../../dataset/wsj0-mix/2speakers/wav16k"

. ./parse_options.sh || exit 1

# Prepare wham noise
file="wham_noise.zip"

if [ -e "${wham_noise_root}/wham_noise/tr/40na010x_1.9857_01xo031a_-1.9857.wav" ] ; then
    echo "Already downloaded dataset ${wham_noise_root}"
else
    if [ ! -d "${wham_noise_root}" ] ; then
        mkdir -p "${wham_noise_root}"
    fi
    wget "https://storage.googleapis.com/whisper-public/${file}" -P "/tmp"
    unzip "/tmp/${file}" -d "/tmp/"
    mv "/tmp/wham_noise/" "${wham_noise_root}"
    rm "/tmp/${file}"
fi

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
    --output-dir "${wham_root}"
elif [ "${create_from}" = "wsjmix" ] ; then
    python create_wham_from_wsjmix.py \
    --wsjmix-dir-8k "${wsjmix_8k}" \
    --wsjmix-dir-16k "${wsjmix_16k}" \
    --wham-noise-root "${wham_noise_root}" \
    --output-dir "${wham_root}"
else
    echo "'create_from' is expected scratch or wsjmix."
fi

cd "${work_dir}"