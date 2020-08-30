#!/bin/bash

out_dir="$1"

echo "download from https://www.itu.int/rec/T-REC-P.862-200102-I/en"

gcc -c dsp.c
gcc -c pesqdsp.c
gcc -c pesqmod.c
gcc -c pesqio.c
gcc -c pesqmain.c

if [ -z ${out_dir} ]; then
    out_dir="."
fi

gcc dsp.o pesqdsp.o pesqmain.o pesqmod.o pesqio.o -o ${out_dir}/PESQ -lm
