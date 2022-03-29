# Conv-TasNetによる話者分離
データセット: LibriSpeech (NOT LibriMix)

## Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/train_librispeech.ipynb)

Google Colaboratoryの環境で試すことができます．`train_librispeech.ipynb`を見てください．

## 実行方法
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time zone.
I recommend that you specify your time zone like `export TZ=UTC-9`.
Here, `TZ=UTC-9` means `Coordinated Universal Time +9 hours`.

### 0. データセットの準備
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/common/
. ./prepare_librispeech.sh \
--librispeech_root <LIBRISPEECH_ROOT> \
--n_sources <#SPEAKERS>
```

### 1. 学習
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./train.sh \
--exp_dir <OUTPUT_DIR>
```

学習を途中から再開したい場合，
```sh
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. 評価
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./test.sh \
--exp_dir <OUTPUT_DIR>
```

### 3. デモ
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./demo.sh
```

## 結果
テストデータに対する評価．
モデルはGoogle Colaboratoryで学習させたものであり，次のようにダウンロードできます．
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./prepare_2speakers-model.sh
```
ネットワーク構造は論文と異なる可能性があります．

| Model | N | L | H | B | Sc | P | X | R | causal | optimizer | lr | SI-SDRi [dB] | PESQ | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 64 | 16 | 256 | 128 | 128 | 3 | 6 | 3 | False | adam | 0.001 |  |  |

# Conv-TasNetによる楽音分離
Dataset: MUSDB18

## 分離例
事前学習済みのConv-TasNetで分離を試すことができます．

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music_ja.ipynb)