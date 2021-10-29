# Open-Unmix (UMX)
参考文献: [Open-Unmix - A Reference Implementation for Music Source Separation](https://hal.inria.fr/hal-02293689/document)

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
cd <REPOSITORY_ROOT>/egs/musdb18/umx/
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--target <TARGET> \
--config_path <CONFIG_PATH>
```

学習を途中から再開したい場合，
```
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH> \
--target <TARGET> \
--config_path <CONFIG_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/musdb18/umx/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 5.02 | 6.06 | 4.00 | 6.14 | 12.18 | 5.30 | 検証ロスが最小となるエポックで学習を止めた場合 |
| UMX | 5.00 | 6.15 | 4.04 | 5.75 | 12.35 | 5.23 | 学習後 |
| UMX-HQ | 4.85 | 5.94 | 4.01 | 6.08 | 12.05 | 5.22 | 検証ロスが最小となるエポックで学習を止めた場合 |
| UMX-HQ | 4.90 | 6.12 | 3.99 | 5.92 | 12.19 | 5.23| 学習後 |
| UMX | 5.23 | 5.73 | 4.02 | 6.32 | - | 5.33 | 公式実装 |
| UMX-HQ | - | - | - | - | - | - | 公式実装 |

- 学習済みモデルを使って分離を試すことができます．`egs/tutorials/umx/separate_music_ja.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_music_ja.ipynb)にとんでください．