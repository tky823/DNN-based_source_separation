#!/bin/bash

out_dir="$1"

# Download

for file in dsp.c pesqdsp.c pesqmod.c pesqio.c pesqmain.c dsp.h pesq.h pesqpar.h ; do
    if [ ! -e ${file} ]; then
        echo "download PESQ software from https://www.itu.int/rec/T-REC-P.862-200102-I/en"
        return
    fi
done

gcc -c dsp.c
gcc -c pesqdsp.c
gcc -c pesqmod.c
gcc -c pesqio.c
gcc -c pesqmain.c

if [ -z ${out_dir} ]; then
    out_dir="."
fi

gcc dsp.o pesqdsp.o pesqmain.o pesqmod.o pesqio.o -o ${out_dir}/PESQ -lm