#!/bin/bash

sr_k=8
minmax='min'
wsj0mix_root="../../dataset/wsj0-mix"

mixed_n_sources='2+3'

. ./parse_options.sh || exit 1

to_dir="${wsj0mix_root}/${mixed_n_sources}speakers/wav${sr_k}k/${minmax}"
mkdir -p "${to_dir}"

n_sources_set=`echo ${mixed_n_sources} | tr '+' '\n'`

for n_sources in ${n_sources_set} ; do
    for data_type in 'tr' 'cv' 'tt' ; do
        from="${wsj0mix_root}/${n_sources}speakers/wav${sr_k}k/${minmax}/${data_type}"
        cp -r "${from}" "${to_dir}"
        
        cat "${wsj0mix_root}/${n_sources}speakers/mix_${n_sources}_spk_${minmax}_${data_type}_mix" >> "${mix_dir}/${mixed_n_sources}speakers/mix_${mixed_n_sources}_spk_${minmax}_${data_type}_mix"
    done
done