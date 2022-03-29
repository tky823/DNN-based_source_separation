# Speech separation using Conv-TasNet
Dataset: LibriSpeech (NOT LibriMix)

## Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/train_librispeech.ipynb)

You can use Google Colaboratory environment. Please see `train_librispeech.ipynb`.

## How to do
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time zone.
I recommend that you specify your time zone like `export TZ=UTC-9`.
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
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
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
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./test.sh \
--exp_dir <OUTPUT_DIR>
```

### 3. Demo
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./demo.sh
```

## Results
Evaluation for test data.
The models are trained on Google Colaboratory, and you can download like this.
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./prepare_2speakers-model.sh
```
Network configuration may be different from original papers.

| Model | N | L | H | B | Sc | P | X | R | causal | optimizer | lr | SI-SDRi [dB] | PESQ | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 64 | 16 | 256 | 128 | 128 | 3 | 6 | 3 | False | adam | 0.001 |  |  |

# Music source separation using Conv-TasNet
Dataset: MUSDB18

## Separation Example
You can separate audio signal by pretrained Conv-TasNet.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music.ipynb)