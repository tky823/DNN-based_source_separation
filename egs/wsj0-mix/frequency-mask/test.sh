#!/bin/bash

exp_dir="./exp"
tag=""

n_sources=2
sr_k=8 # sr_k=8 means sampling rate is 8kHz. Choose from 8kHz or 16kHz.
sr=${sr_k}000
duration=4
max_or_min='min'

test_wav_root="../../../dataset/wsj0-mix/${n_sources}speakers/wav${sr_k}k/${max_or_min}/tt"
test_list_path="../../../dataset/wsj0-mix/${n_sources}speakers/mix_${n_sources}_spk_${max_or_min}_tt_mix"

# STFT and masking
window_fn='hann'
fft_size=256
hop_size=64
ideal_mask='ibm'

# Criterion
criterion='sisdr'

overwrite=0
gpu_id="0"

. ./path.sh
. parse_options.sh || exit 1

if [ -z "${tag}" ]; then
    save_dir="${exp_dir}/${ideal_mask}/${n_sources}mix/sr${sr_k}k_${max_or_min}/${criterion}/stft${fft_size}-${hop_size}_${window_fn}-window"
else
    save_dir="${exp_dir}/${tag}"
fi

log_dir="${save_dir}/log"
out_dir="${save_dir}/test"

if [ ! -e "${log_dir}" ]; then
    mkdir -p "${log_dir}"
fi

time_stamp=`date "+%Y%m%d-%H%M%S"`

export CUDA_VISIBLE_DEVICES="${gpu_id}"

test.py \
--test_wav_root ${test_wav_root} \
--test_list_path ${test_list_path} \
--sr ${sr} \
--window_fn ${window_fn} \
--fft_size ${fft_size} \
--hop_size ${hop_size} \
--method ${ideal_mask} \
--n_sources ${n_sources} \
--criterion ${criterion} \
--out_dir "${out_dir}" \
--overwrite ${overwrite} | tee "${log_dir}/test_${time_stamp}.log"
