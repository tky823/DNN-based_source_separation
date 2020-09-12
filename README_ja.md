# DNNによる音源分離
DNNによる音源分離（PyTorch実装）

## モデル
| モデル | 参考文献 | 実装 |
| :---: | :---: | :---: |
| WaveNet | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | ✔ |
| Wave-U-Net | [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185) |  |
| Deep clustering | [Single-Channel Multi-Speaker Separation using Deep Clustering](https://arxiv.org/abs/1607.02173) |  |
| Chimera++ | [Alternative Objective Functions for Deep Clustering](https://www.merl.com/publications/docs/TR2018-005.pdf) |  |
| DANet | [Deep attractor network for single-microphone speaker separation](https://arxiv.org/abs/1611.08930) | ✔ |
| ADANet | [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634) |  |
| TasNet | [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) | ✔ |
| Conv-TasNet | [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) | ✔ |
| DPRNN-TasNet | [Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation](https://arxiv.org/abs/1910.06379) | ✔ |
| Gated DPRNN-TasNet | [Voice Separation with an Unknown Number of Multiple Speakers](https://arxiv.org/abs/2003.01531) |  |
| DeepCASA | [Divide and conquer: A deep casa approach to talker-independent monaural speaker separation](https://arxiv.org/abs/1904.11148) |  |
| FurcaNet | [FurcaNet: An end-to-end deep gated convolutional, long short-term memory, deep neural networks for single channel speech separation](https://arxiv.org/abs/1902.00651) |  |
| Wavesplit | [Wavesplit: End-to-End Speech Separation by Speaker Clustering](https://arxiv.org/abs/2002.08933) |  |

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
- v0.0.0: Conv-TasNetとDPRNN-TasNetによるLibriSpeechデータセットの例を含む．
- v0.0.1: データセット名の変更．
- v0.1.0: データセットの構造の変更．
- v0.1.1: DANetを追加．
- v0.1.2: Conv-TasNetのレーヤー名の変更．DANetの入力特徴量をlog-magnitudeに変更．
