# DNNによる音源分離
DNNによる音源分離（PyTorch実装）

## モデル
| モデル | 参考文献 | 実装 |
| :---: | :---: | :---: |
| WaveNet | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | ✔ |
| Wave-U-Net | [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185) |  |
| Deep clustering | [Single-Channel Multi-Speaker Separation using Deep Clustering](https://arxiv.org/abs/1607.02173) |  |
| Chimera++ | [Alternative Objective Functions for Deep Clustering](https://www.merl.com/publications/docs/TR2018-005.pdf) |  |
| DANet | [Deep Attractor Network for Single-microphone Apeaker Aeparation](https://arxiv.org/abs/1611.08930) | ✔ |
| ADANet | [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634) |  |
| TasNet | [TasNet: Time-domain Audio Separation Network for Real-time, Single-channel Speech Separation](https://arxiv.org/abs/1711.00541) | ✔ |
| Conv-TasNet | [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) | ✔ |
| DPRNN-TasNet | [Dual-path RNN: Efficient Long Sequence Modeling for Time-domain Single-channel Speech Separation](https://arxiv.org/abs/1910.06379) | ✔ |
| Gated DPRNN-TasNet | [Voice Separation with an Unknown Number of Multiple Speakers](https://arxiv.org/abs/2003.01531) |  |
| FurcaNet | [FurcaNet: An End-to-End Deep Gated Convolutional, Long Short-term Memory, Deep Neural Networks for Single Channel Speech Separation](https://arxiv.org/abs/1902.00651) |  |
| FurcaNeXt | [FurcaNeXt: End-to-End Monaural Speech Separation with Dynamic Gated Dilated Temporal Convolutional Networks](https://arxiv.org/abs/1902.04891) |
| DeepCASA | [Divide and Conquer: A Deep Casa Approach to Talker-independent Monaural Speaker Separation](https://arxiv.org/abs/1904.11148) |  |
| Wavesplit | [Wavesplit: End-to-End Speech Separation by Speaker Clustering](https://arxiv.org/abs/2002.08933) |  |
| DPTNet | [Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation](https://arxiv.org/abs/2007.13975) | ✔ |
| D3Net | [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733) | ✔ |
| SepFormer | [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154) |  |
| GALR | [Effective Low-Cost Time-Domain Audio Separation Using Globally Attentive Locally Reccurent networks](https://arxiv.org/abs/2101.05014) | ✔ |

## 学習に関する方法
| 方法 | 参考文献 | 実装 |
| :---: | :---: | :---: |
| Pemutation invariant training (PIT) | [Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of Deep Recurrent Neural Networks](https://arxiv.org/abs/1703.06284) | ✔ |
| One-and-rest PIT | [Recursive Speech Separation for Unknown Number of Speakers](https://arxiv.org/abs/1904.03065) | ✔ |

## 実行例
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/train_conv-tasnet.ipynb)

[Conv-TasNet](https://arxiv.org/abs/1809.07454)によるLibriSpeechデータセットを用いた音源分離の例

`<REPOSITORY_ROOT>/egs/tutorials/`で他のチュートリアルも確認可能．

### 0. データセットの準備
```
cd <REPOSITORY_ROOT>/egs/tutorials/common/
. ./prepare_librispeech.sh --dataset_root <DATASET_DIR> --n_sources <#SPEAKERS>
```

### 1. 学習
```
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./train.sh --exp_dir <OUTPUT_DIR>
```

学習を途中から再開したい場合，
```
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

### 3. デモンストレーション
```
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./demo.sh
```

## バージョン情報
- v0.0.0: Conv-TasNetとDPRNN-TasNetによるLibriSpeechデータセットの例を含む．
- v0.0.1: データセット名の変更．
- v0.1.0: データセットの構造の変更．
- v0.1.1: DANetを追加．
- v0.1.2: Conv-TasNetのレイヤー名の変更．DANetの入力特徴量をlog-magnitudeに変更．
- v0.1.3: Wall Street Journal 0 (WSJ0) データセット用のスクリプトを追加．
- v0.1.4: 非負値行列因子分解 (non-nagative matrix factorization; NMF)を追加．
- v0.2.0: 短時間フーリエ変換の表現を変更．
- v0.2.1: `conv_tasnet`ディレクトリの名前を`conv-tasnet`に変更．ORPIT (one-and-rest PIT)を追加．
- v0.3.0: `wsj0`を`wsj0-mix`へ名前を変更．実験結果の更新．
- v0.3.1: TasNetにおける線形のencoderに対応．
- v0.3.2: dual-path RNNのチャネル数の定義を変更．
- v0.3.3: v0.3.2の影響により，学習済みモデルを更新．
- v0.4.0: DPRNN-TasNetのネットワーク構造を変更．
- v0.4.1: DPTNetおよびGALRNetを追加．DPRNN-TasNetを再修正．
- v0.4.2: GALRNet用の学習スクリプトを追加．
- v0.4.3: DPRNN-TasNetを再修正．
- v0.5.0: `parse_options.sh`を追加．
- v0.5.1: 一部のモデルで多チャネルの入力に対応．
- v0.5.2: 距離学習に関するチュートリアルを追加．