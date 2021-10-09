#!/bin/bash

vctk_root="../../../dataset/VCTK"

. ./parse_options.sh || exit 1

file="VCTK-Corpus.tar.gz"

if [ -d "${vctk_root}" ] ; then
    echo "Already downloaded VCTK dataset."
else
    if [ ! -d "${vctk_root}" ] ; then
        mkdir -p "${vctk_root}"
    fi
    wget "http://www.udialogue.org/download/${file}" -P "/tmp/"
    tar -xzvf "/tmp/${file}" -C "/tmp/"
    rm "/tmp/${file}"

    mv "/tmp/VCTK-Corpus/"* "${vctk_root}"
fi