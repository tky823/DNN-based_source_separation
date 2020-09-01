# LibriSpeech

## How to do
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time zone.
I recommend that you specify your time zone like `TZ=UTC-9 date "+%Y%m%d-%H%M%S"`.
Here, `TZ=UTC-9` means `Coordinated Universal Time +9 hours`.

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
Models are placed in sub-directory. These models are trained on Google Colaboratory.
Network configuration may be different from original papers.

| Model | SI-SDRi [dB] | PESQ |
| :---: | :---: | :---: |
| DANet |  |  |
| ADANet |  |  |
| TasNet |  |  |
| Conv-TasNet |  |  |
| DPRNN-TasNet |  |  |
