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

| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 5.71 | 6.00 | 4.82 | 3.99 | 12.14 | 5.13 | 検証ロスが最小となるエポックで学習を止めた場合． |
| UMX | 5.81 | 6.09 | 4.69 | 3.66 | 12.07 | 5.06 | 100エポック学習後 |
| UMX | 6.32 | 5.73 | 5.23 | 4.02 | - | 5.33 | 公式実装 |

- 学習済みモデルを使って分離を試すことができます．`egs/tutorials/umx/separate_ja.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_ja.ipynb)にとんでください．