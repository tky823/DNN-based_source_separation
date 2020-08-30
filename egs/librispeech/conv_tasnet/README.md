# LibriSpeech using Conv-TasNet

### 0. Preparation
```
cd <REPOSITORY_ROOT>/egs/librispeech/common/
. ./prepare.sh <DATASET_DIR> <#SPEAKERS>
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./train.sh <OUTPUT_DIR>
```

If you want to resume training,
```
. ./train.sh <OUTPUT_DIR> <MODEL_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./test.sh <OUTPUT_DIR>
```

### 3. Demo
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./demo.sh
```

## Results
Evaluation for test data.
Models are placed in sub-directory. These models are train on Google Colaboratory.
Network configuration may be different from original papers.

| Model | N | L | H | B | Sc | P | X | R | causal | SI-SDRi [dB] | PESQ | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 64 | 16 | 256 | 128 | 128 | 3 | 6 | 3 | True |  |  |
