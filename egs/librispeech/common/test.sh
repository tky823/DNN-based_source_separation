#!/bin/bash

echo "download from https://www.itu.int/rec/T-REC-P.862-200102-I/en"

gcc -c dsp.c
gcc -c pesqdsp.c
gcc -c pesqmod.c
gcc -c pesqio.c
gcc -c pesqmain.c

gcc dsp.o pesqdsp.o pesqmain.o pesqmod.o pesqio.o -o PESQ -lm
