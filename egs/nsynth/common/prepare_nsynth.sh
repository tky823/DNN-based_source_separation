#!/bin/bash

nsynth_root="../../../dataset/nsynth"
train_url="http://download.magenta.tensorflow.org/datasets/nsynth"
train_dataset="nsynth-train.jsonwav.tar.gz"
valid_url="http://download.magenta.tensorflow.org/datasets/nsynth"
valid_dataset="nsynth-valid.jsonwav.tar.gz"
test_url="http://download.magenta.tensorflow.org/datasets/nsynth"
test_dataset="nsynth-test.jsonwav.tar.gz"

if [ -e "${nsynth_root}/${train_dataset}" ]; then
    echo "Already downloaded dataset ${train_dataset}"
else
    wget "${train_url}/${train_dataset}" -P "/tmp"
    tar -zxvf "/tmp/${train_dataset}" -C "${nsynth_root}"
fi

if [ -e "${nsynth_root}/${valid_dataset}" ]; then
    echo "Already downloaded dataset ${valid_dataset}"
else
    wget "${valid_url}/${valid_dataset}" -P "/tmp"
    tar -zxvf "/tmp/${valid_dataset}" -C "${nsynth_root}"
fi

if [ -e "${nsynth_root}/${test_dataset}" ]; then
    echo "Already downloaded dataset ${test_dataset}"
else
    wget "${test_url}/${test_dataset}" -P "/tmp"
    tar -zxvf "/tmp/${test_dataset}" -C "${nsynth_root}"
fi