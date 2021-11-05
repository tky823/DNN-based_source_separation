# CrossNet-Open-Unmix (X-UMX)

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
cd <REPOSITORY_ROOT>/egs/musdb18/x-umx/
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--config_path <CONFIG_PATH>
```

学習を途中から再開したい場合，
```
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH> \
--config_path <CONFIG_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/musdb18/x-umx/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-UMX | 4.84 | 6.01 | 3.75 | 5.53 | 12.10 | 5.03 | 検証ロスが最小となるエポックで学習を止めた場合 |
| X-UMX | 4.47 | 5.77 | 3.53 | 5.53 | 11.93 | 4.82 | 学習後 |
| X-UMX | - | - | - | - | - | - | 公式実装 |

- 学習済みモデルを使って分離を試すことができます．`egs/tutorials/x-umx/separate_music_ja.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/x-umx/separate_music_ja.ipynb)にとんでください．