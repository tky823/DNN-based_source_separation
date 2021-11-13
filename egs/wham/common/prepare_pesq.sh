#!/bin/bash

out_dir=""

. ./parse_options.sh

# Download
files=( "dsp.c" "pesqdsp.c" "pesqmod.c" "pesqio.c" "pesqmain.c" "dsp.h" "pesq.h" "pesqpar.h" )
for file in "${files[@]}" ; do
    if [ ! -e ${file} ]; then
        echo "download PESQ software from https://www.itu.int/rec/T-REC-P.862-200102-I/en"
        return
    fi
done

files=( "dsp.c" "pesqdsp.c" "pesqmod.c" "pesqio.c" "pesqmain.c" )
for file in "${files[@]}" ; do
    gcc -c ${file}
done

if [ -z ${out_dir} ]; then
    out_dir="./"
fi

gcc dsp.o pesqdsp.o pesqmain.o pesqmod.o pesqio.o -o ${out_dir}/PESQ -lm