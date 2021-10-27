# Conv-TasNet
## 実行方法
### 0. データセットの準備
pipによる環境構築
```
cd <REPOSITORY_ROOT>/egs/musdb18/
pip install -r requirements.txt
```
or condaによる環境構築
```
cd <REPOSITORY_ROOT>/egs/musdb18/
conda env create -f environment-gpu.yaml
```

MUSDB18データセットのダウンロードと`.wav`への変換
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18_root <MUSDB18_ROOT> \
--is_hq 0 \
--to_wav 1
```
MUSDB18-HQデータセットを用いる場合，
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18_root <MUSDB18HQ_ROOT> \
--is_hq 1
```

### 1. 学習
```
cd <REPOSITORY_ROOT>/egs/musdb18/conv-tasnet/
. ./train.sh \
--exp_dir <OUTPUT_DIR>
```

学習を途中から再開したい場合，
```
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/musdb18/conv-tasnet/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 5.32 | 6.06 | 4.00 | 6.04 | 12.33 | 5.35 | - |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.95 | 6.11 | 3.78 | 5.59 | 11.90 | 5.36 | - |
| Conv-TasNet | 44100 | 8 | 64 | 256 | 512 | 5.38 | 5.82 | 3.51 | 5.91 | 11.85 | 5.16 | - |
| Conv-TasNet | 44100 | - | 20 | - | - | 5.66 | 6.08 | 4.37 | 6.81 | - | 5.73 | Official report. Network architecture is different from one in this repo. |

- 学習済みモデルを使って分離を試すことができます．`egs/tutorials/conv-tasnet/separate_music_ja.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music_ja.ipynb)にとんでください．