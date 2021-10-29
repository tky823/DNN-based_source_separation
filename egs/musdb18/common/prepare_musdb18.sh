#!/bin/bash

musdb18_root="" # By default musdb18_root="../../../dataset/MUSDB18" if is_hq=0, musdb18_root="../../../dataset/MUSDB18HQ" otherwise.
is_7s=0
is_hq=0
to_wav=0

subsets=( "train" "test" )

. ./parse_options.sh || exit 1

if [ ${is_hq} -eq 0 ]; then
    if [ ${is_7s} -eq 0 ]; then
        file="musdb18.zip"
        url="https://zenodo.org/record/1117372/files/${file}"
        if [ -z "${musdb18_root}" ]; then
            musdb18_root="../../../dataset/MUSDB18-7s"
        fi
    else
        file="MUSDB18-7-STEMS.zip"
        url="https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/${file}"
        
        if [ -z "${musdb18_root}" ]; then
            musdb18_root="../../../dataset/MUSDB18"
        fi
    fi
    
    if [ -e "${musdb18_root}/train/A Classic Education - NightOwl.stem.mp4" ]; then
        echo "Already downloaded dataset ${musdb18_root}"
    else
        mkdir -p "${musdb18_root}"
        wget "${url}" -P "/tmp"
        unzip "/tmp/${file}" -d "${musdb18_root}"
        rm "/tmp/${file}"
    fi

    if [ ${to_wav} -eq 1 ]; then
        if [ -e "${musdb18_root}/train/A Classic Education - NightOwl/" ]; then
            echo "Dataset has been already converted"
        else
            # Reference: https://github.com/sigsep/sigsep-mus-io/blob/master/scripts/decode.sh
            
            work_dir="$PWD"
            cd "${musdb18_root}"

            for t in "${subsets[@]}" ; do
                cd $t
                for stem in *.stem.mp4 ; do
                    name=`echo $stem | awk -F".stem.mp4" '{$0=$1}1'`;
                    echo "$stem"
                    mkdir "$name"
                    cd "$name"
                    ffmpeg -loglevel panic -i "../${stem}" -map 0:0 -vn mixture.wav
                    ffmpeg -loglevel panic -i "../${stem}" -map 0:1 -vn drums.wav
                    ffmpeg -loglevel panic -i "../${stem}" -map 0:2 -vn bass.wav
                    ffmpeg -loglevel panic -i "../${stem}" -map 0:3 -vn other.wav
                    ffmpeg -loglevel panic -i "../${stem}" -map 0:4 -vn vocals.wav
                    cd ../
                done
                cd ../
            done
            cd "${work_dir}"
        fi
    else
        echo "Set --to_wav 1 to load audio in this project."
    fi
else
    file=musdb18hq.zip
    if [ -z "${musdb18_root}" ]; then
        musdb18_root="../../../dataset/MUSDB18HQ"
    fi

    if [ -e "${musdb18_root}/train/A Classic Education - NightOwl/vocals.wav" ]; then
        echo "Already downloaded dataset ${musdb18_root}"
    else
        mkdir -p "${musdb18_root}"
        wget "https://zenodo.org/record/3338373/files/${file}" -P "/tmp"
        unzip "/tmp/${file}" -d "${musdb18_root}"
        rm "/tmp/${file}"
    fi
fi

subsets=( "train" "validation" "test" )

for subset in "${subsets[@]}" ; do
    if [ ! -e "${musdb18_root}/${subset}.txt" ]; then
        if [ ${is_7s} -eq 0 ]; then
            cp "../../../dataset/MUSDB18/${subset}.txt" "${musdb18_root}"
        else
            cp "../../../dataset/MUSDB18-7s/${subset}.txt" "${musdb18_root}"
        fi
    fi
done