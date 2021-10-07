# WHAM!データセットを用いた音源分離
## データセットの作成方法
### 1. wsj0-mixの用意
WSJ0データセットをダウンロードし，wsj0-2mixデータセットを作成する．

### 2. WHAM noiseのダウンロードとWHAM!データセットの作成
```
cd <REPOSITORY_ROOT>/egs/wham/common/
. ./prepare_wham.sh \
--wham_root <WHAM_ROOT> \
--wham_noise_root <WHAM_NOISE_ROOT> \
--wsjmix_2speakers_8k <WSJ0-MIX_2speakers_8k_ROOT> \
--wsjmix_2speakers_16k <WSJ0-MIX_2speakers_16k_ROOT> \
--create_from "wsjmix"
```
WSJ0の元のデータから作成したい場合，
```
cd <REPOSITORY_ROOT>/egs/wham/common/
. ./prepare_wham.sh \
--wham_root <WHAM_ROOT> \
--wham_noise_root <WHAM_NOISE_ROOT> \
--wsj0_root <WSJ0_ROOT> \
--create_from "scratch"
```

## DNNの学習と評価
各サブディレクトリを参照．

## 結果
