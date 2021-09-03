# D3Net
Reference: [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733)

## 実行方法
### 0. データセットの準備
pipによる環境構築
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
pip install -r requirements.txt
```
or condaによる環境構築
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
conda env create -f environment-gpu.yaml
```

MUSDB18データセットのダウンロードと`.wav`への変換
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18_root <MUSDB18_ROOT> \
--to_wav 1
```
MUSDB18-HQデータセットを用いる場合，
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18hq_root <MUSDB18HQ_ROOT> \
--is_hq 1
```

### 1. 学習
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
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
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## 実験結果
SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - | - |
| D3Net | 7.02 | 6.58 | 4.88 | 4.77 | 13.38 | 5.81 | 検証損失によって選択されたエポックで学習を止めた場合 |
| D3Net | 7.08 | 6.54 | 4.93 | 4.72 | 13.41 | 5.82 | 50エポック後 |
| D3Net | 7.24 | 7.01 | 5.25 | 4.53 | 13.52 | 6.01 | 公式実装 |