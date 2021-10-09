# LibriSpeech separation using DANet

## Google Colaboratory
You can use Google Colaboratory environment. Please see `train_danet.ipynb`

## How to do
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time zone.
I recommend that you specify your time zone like `TZ=UTC-9 date "+%Y%m%d-%H%M%S"`.
Here, `TZ=UTC-9` means `Coordinated Universal Time +9 hours`.

### 0. Preparation
```
cd <REPOSITORY_ROOT>/egs/librispeech/common/
. ./prepare_librispeech.sh \
--dataset_root <DATASET_DIR> \
--n_sources <#SPEAKERS>
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/librispeech/danet/
. ./train.sh --exp_dir <OUTPUT_DIR>
```

If you want to resume training,
```
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/librispeech/danet/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

### 3. Demo
```
cd <REPOSITORY_ROOT>/egs/librispeech/danet/
. ./demo.sh
```

## Results
Evaluation for test data.
