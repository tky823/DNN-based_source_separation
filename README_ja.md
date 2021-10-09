# DNNによる音源分離
DNNによる音源分離（PyTorch実装）

## 新しい情報
- v0.6.1: モジュールの追加．

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
| Conditioned-U-Net | [Conditioned-U-Net: Introducing a Control Mechanism in the U-Net for multiple source separations](https://arxiv.org/abs/1907.01277) | ✔ |
| UMX (Open-Unmix) | [Open-Unmix - A Reference Implementation for Music Source Separation](https://hal.inria.fr/hal-02293689/document) | ✔ |
| Wavesplit | [Wavesplit: End-to-End Speech Separation by Speaker Clustering](https://arxiv.org/abs/2002.08933) |  |
| DPTNet | [Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation](https://arxiv.org/abs/2007.13975) | ✔ |
| D3Net | [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733) | ✔ |
| LaSAFT | [LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation](https://arxiv.org/abs/2010.11631) |  |
| SepFormer | [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154) |  |
| GALR | [Effective Low-Cost Time-Domain Audio Separation Using Globally Attentive Locally Reccurent networks](https://arxiv.org/abs/2101.05014) | ✔ |

## モジュール
| モジュール | 参考文献 | 実装 |
| :---: | :---: | :---: |
| Depthwise-separable convolution |  | ✔ |
| Gated Linear Units |  | ✔ |
| FiLM (Feature-wise Linear Modulation) | [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) | ✔ |
| PoCM (Point-wise Convolutional Modulation) | [LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation](https://arxiv.org/abs/2010.11631) | ✔ |

## 学習に関する方法
| 方法 | 参考文献 | 実装 |
| :---: | :---: | :---: |
| Pemutation invariant training (PIT) | [Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of Deep Recurrent Neural Networks](https://arxiv.org/abs/1703.06284) | ✔ |
| One-and-rest PIT | [Recursive Speech Separation for Unknown Number of Speakers](https://arxiv.org/abs/1904.03065) | ✔ |
| Probabilistic PIT | [Probabilistic Permutation Invariant Training for Speech Separation](https://arxiv.org/abs/1908.01768) |  |
| Sinkhorn PIT | [Towards Listening to 10 People Simultaneously: An Efficient Permutation Invariant Training of Audio Source Separation Using Sinkhorn's Algorithm](https://arxiv.org/abs/2010.11871) | ✔ |

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
