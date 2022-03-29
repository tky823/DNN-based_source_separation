# LibriSpeech separation using DANet

## Google Colaboratory
You can use Google Colaboratory environment. Please see `train_danet.ipynb`

## How to do
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time zone.
I recommend that you specify your time zone like `TZ=UTC-9 date "+%Y%m%d-%H%M%S"`.
Here, `TZ=UTC-9` means `Coordinated Universal Time +9 hours`.

### 0. Preparation
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/common/
. ./prepare_librispeech.sh \
--librispeech_root <LIBRISPEECH_ROOT> \
--n_sources <#SPEAKERS>
```

### 1. Training
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/danet/
. ./train.sh \
--exp_dir <OUTPUT_DIR>
```

If you want to resume training,
```sh
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. Evaluation
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/danet/
. ./test.sh \
--exp_dir <OUTPUT_DIR>
```

### 3. Demo
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/danet/
. ./demo.sh
```

## Results
Evaluation for test data.
