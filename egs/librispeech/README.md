# LibriSpeech

### 0. Preparation
```
cd <REPOSITORY_ROOT>/egs/librispeech/common/
. ./prepare.sh <DATASET_DIR> <#SPEAKERS>
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/librispeech/<MODEL_NAME>/
. ./train.sh <OUTPUT_DIR>
```

If you want to resume training,
```
. ./train.sh <OUTPUT_DIR> <MODEL_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/librispeech/<MODEL_NAME>/
. ./test.sh <OUTPUT_DIR>
```

### 3. Demo
```
cd <REPOSITORY_ROOT>/egs/librispeech/<MODEL_NAME>/
. ./demo.sh
```

## Results
Evaluation for test data.
Models are placed in sub-directory. These models are train on Google Colaboratory.
Network configuration may be different from original papers.

| Model | SI-SDRi [dB] | PESQ |
| :---: | :---: | :---: |
| TasNet |  |  |
| Conv-TasNet |  |  |
| DPRNN-TasNet |  |  |
