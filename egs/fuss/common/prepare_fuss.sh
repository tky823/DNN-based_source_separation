#!/bin/bash

fuss_root="../../../dataset/FUSS"

. ./parse_options.sh || exit 1

if [ ! -d "${fuss_root}" ]; then
    mkdir -p "${fuss_root}"
fi

if [ ! -d "${fuss_root}/ssdata" ]; then
    file="FUSS_ssdata.tar.gz"
    wget "https://zenodo.org/record/3743844/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "${fuss_root}"
    rm "/tmp/${file}"
fi

if [ ! -d "${fuss_root}/ssdata_reverb" ]; then
    file="FUSS_ssdata_reverb.tar.gz"
    wget "https://zenodo.org/record/3743844/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "${fuss_root}"
    rm "/tmp/${file}"
fi

if [ ! -d "${fuss_root}/fsd_data" ]; then
    file="FUSS_fsd_data.tar.gz"
    wget "https://zenodo.org/record/3743844/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "${fuss_root}"
    rm "/tmp/${file}"
fi

if [ ! -d "${fuss_root}/rir_data" ]; then
    file="FUSS_rir_data.tar.gz"
    wget "https://zenodo.org/record/3743844/files/${file}" -P "/tmp"
    tar -zxvf "/tmp/${file}" -C "${fuss_root}"
    rm "/tmp/${file}"
fi