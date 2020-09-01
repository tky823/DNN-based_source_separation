# DNNによる音源分離
DNNによる音源分離（PyTorch実装）

## モデル
| モデル | 参考文献 |
| :---: | :---: |
| WaveNet | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) |
| Deep clustering | [Single-Channel Multi-Speaker Separation using Deep Clustering](https://arxiv.org/abs/1607.02173) |
| DANet | [Deep attractor network for single-microphone speaker separation](https://arxiv.org/abs/1611.08930) |
| ADANet | [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634) |
| TasNet | [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) |
| Conv-TasNet | [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) |
| DPRNN-TasNet | [Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation](https://arxiv.org/abs/1910.06379) |

## 実行例
[Conv-TasNet](https://arxiv.org/abs/1809.07454)によるLibriSpeechデータセットを用いた音源分離の例
```
cd <REPOSITORY_ROOT>/egs/librispeech/
```

### 0. データセットの準備
```
cd <REPOSITORY_ROOT>/egs/librispeech/common/
. ./prepare.sh <DATASET_DIR> <#SPEAKERS>
```

### 1. 学習
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./train.sh <OUTPUT_DIR>
```

学習を途中から再開したい場合，
```
. ./train.sh <OUTPUT_DIR> <MODEL_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./test.sh <OUTPUT_DIR>
```

### 3. デモンストレーション
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./demo.sh
```

## バージョン情報
v0.0.0: Conv-TasNetとDPRNN-TasNetによるLibriSpeechデータセットの例を含む．

v0.0.1: データセット名の変更．

v0.1.0: データセットの構造の変更．
