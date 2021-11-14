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

## 結果
### 1話者音声強調

| Model | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: |
| LSTM-TasNet | - | - | - |
| Conv-TasNet | 13.8 | 14.2 | 2.93 |

### 2話者音声強調

| Model | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: |
| LSTM-TasNet | - | - | - |
| Conv-TasNet | 12.7 | 13.1 | 2.39 |

### 雑音環境下の2話者分離の結果．

| モデル | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: |
| LSTM-TasNet | - | - | - |
| Conv-TasNet | 12.3 | 12.7 | 2.36 |