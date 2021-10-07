#!/bin/bash

goodsounds_root="../../../dataset/good-sounds"

. ./parse_options.sh || exit 1

file="good-sounds.zip"

if [ -d "${goodsounds_root}" ] ; then
    echo "Already downloaded GOOD-SOUNDS dataset."
else
    if [ ! -d "${goodsounds_root}" ] ; then
        mkdir -p "${goodsounds_root}"
    fi
    wget "https://zenodo.org/record/4588740/files/${file}" -P "/tmp/"
    unzip "/tmp/${file}" -d "/tmp/"
    rm "/tmp/${file}"

    mv "/tmp/good-sounds/"* "${goodsounds_root}"
fi