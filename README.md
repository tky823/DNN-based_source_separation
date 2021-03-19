# DNN-based source separation
A PyTorch implementation of DNN-based source separation.

## Model
| Model | Reference | Done |
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
| D3Net | [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733) |
| GALR | [Effective Low-Cost Time-Domain Audio Separation Using Globally Attentive Locally Reccurent networks](https://arxiv.org/abs/2101.05014) | ✔ |

## Method related to training
| Method | Reference | Done |
| :---: | :---: | :---: |
| Pemutation invariant training (PIT) | [Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of Deep Recurrent Neural Networks](https://arxiv.org/abs/1703.06284) | ✔ |
| One-and-rest PIT | [Recursive Speech Separation for Unknown Number of Speakers](https://arxiv.org/abs/1904.03065) | ✔ |

## Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/master/egs/librispeech/conv-tasnet/train_conv-tasnet.ipynb)

LibriSpeech example using [Conv-TasNet](https://arxiv.org/abs/1809.07454)
```
cd <REPOSITORY_ROOT>/egs/librispeech/
```

### 0. Preparation
```
cd <REPOSITORY_ROOT>/egs/librispeech/common/
. ./prepare.sh <DATASET_DIR> <#SPEAKERS>
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv-tasnet/
. ./train.sh --exp_dir <OUTPUT_DIR>
```

If you want to resume training,
```
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv-tasnet/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

### 3. Demo
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv-tasnet/
. ./demo.sh
```

## Version
- v0.0.0: Initial version. LibriSpeech Conv-TasNet & DPRNN-TasNet examples are included.
- v0.0.1: Dataset is renamed.
- v0.1.0: Dataset structure is changed.
- v0.1.1: DANet is included.
- v0.1.2: Layer name is changed. Input feature for DANet is replaced by log-magnitude.
- v0.1.3: Add scripts for Wall Street Journal 0 (WSJ0) dataset.
- v0.1.4: Add non-nagative matrix factorization (NMF).
- v0.2.0: Change the representation of short time Fourier transform (STFT).
- v0.2.1: `conv_tasnet` directory is renamed to `conv-tasnet`. Add one-and-rest PIT (ORPIT).
- v0.3.0: `wsj0` is renamed to `wsj0-mix`. The result is updated.
- v0.3.1: Implement Linear encoder for TasNet.
- v0.3.2: Change the definition of `hidden_channels` in dual-path RNN.
- v0.3.3: Fix trained models due to the update v0.3.2.
- v0.4.0: Fix the network architecture of DPRNN-TasNet.
- v0.4.1: Add DPTNet and GALRNet. Re-fix DPRNN-TasNet.
- v0.4.2: Add training script for GALRNet.
- v0.4.3: Re-fix DPRNN-TasNet.
- v0.5.0: Add `parse_options.sh`.
- v0.5.1: Multichannel support.