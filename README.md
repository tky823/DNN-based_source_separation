# DNN-based source separation
A PyTorch implementation of DNN-based source separation.

## New information
- v0.7.2
  - Update jupyter notebooks.

## Model
| Model | Reference | Done |
| :---: | :---: | :---: |
| WaveNet | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | ✔ |
| Wave-U-Net | [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185) |  |
| Deep Clustering | [Deep Clustering: Discriminative Embeddings for Segmentation and Separation](https://arxiv.org/abs/1508.04306) | ✔ |
| Deep Clustering++ | [Single-Channel Multi-Speaker Separation using Deep Clustering](https://arxiv.org/abs/1607.02173) |  |
| Chimera | [Alternative Objective Functions for Deep Clustering](https://www.merl.com/publications/docs/TR2018-005.pdf) |  |
| DANet | [Deep Attractor Network for Single-microphone Apeaker Aeparation](https://arxiv.org/abs/1611.08930) | ✔ |
| ADANet | [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634) | ✔ |
| TasNet | [TasNet: Time-domain Audio Separation Network for Real-time, Single-channel Speech Separation](https://arxiv.org/abs/1711.00541) | ✔ |
| Conv-TasNet | [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) | ✔ |
| DPRNN-TasNet | [Dual-path RNN: Efficient Long Sequence Modeling for Time-domain Single-channel Speech Separation](https://arxiv.org/abs/1910.06379) | ✔ |
| Gated DPRNN-TasNet | [Voice Separation with an Unknown Number of Multiple Speakers](https://arxiv.org/abs/2003.01531) |  |
| FurcaNet | [FurcaNet: An End-to-End Deep Gated Convolutional, Long Short-term Memory, Deep Neural Networks for Single Channel Speech Separation](https://arxiv.org/abs/1902.00651) |  |
| FurcaNeXt | [FurcaNeXt: End-to-End Monaural Speech Separation with Dynamic Gated Dilated Temporal Convolutional Networks](https://arxiv.org/abs/1902.04891) |
| DeepCASA | [Divide and Conquer: A Deep Casa Approach to Talker-independent Monaural Speaker Separation](https://arxiv.org/abs/1904.11148) |  |
| Conditioned-U-Net | [Conditioned-U-Net: Introducing a Control Mechanism in the U-Net for multiple source separations](https://arxiv.org/abs/1907.01277) | ✔ |
| MMDenseNet | [Multi-scale Multi-band DenseNets for Audio Source Separation](https://arxiv.org/abs/1706.09588) | ✔ |
| MMDenseLSTM | [MMDenseLSTM: An Efficient Combination of Convolutional and Recurrent Neural Networks for Audio Source Separation](https://arxiv.org/abs/1805.02410) | ✔ |
| Open-Unmix (UMX) | [Open-Unmix - A Reference Implementation for Music Source Separation](https://hal.inria.fr/hal-02293689/document) | ✔ |
| Wavesplit | [Wavesplit: End-to-End Speech Separation by Speaker Clustering](https://arxiv.org/abs/2002.08933) |  |
| Hydranet | [Hydranet: A Real-Time Waveform Separation Network](https://ieeexplore.ieee.org/document/9053357) |  |
| Dual-Path Transformer Network (DPTNet) | [Dual-Path Transformer Network: Direct Context-Aware Modeling for End-to-End Monaural Speech Separation](https://arxiv.org/abs/2007.13975) | ✔ |
| CrossNet-Open-Unmix (X-UMX) | [All for One and One for All: Improving Music Separation by Bridging Networks](https://arxiv.org/abs/2010.04228) | ✔ |
| D3Net | [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733) | ✔ |
| LaSAFT | [LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation](https://arxiv.org/abs/2010.11631) |  |
| SepFormer | [Attention is All You Need in Speech Separation](https://arxiv.org/abs/2010.13154) | ✔ |
| GALR | [Effective Low-Cost Time-Domain Audio Separation Using Globally Attentive Locally Reccurent networks](https://arxiv.org/abs/2101.05014) | ✔ |
| HRNet | [Vocal Melody Extraction via HRNet-Based Singing Voice Separation and Encoder-Decoder-Based F0 Estimation](https://www.mdpi.com/2079-9292/10/3/298) | ✔ |
| MRX | [The Cocktail Fork Problem: Three-Stem Audio Separation for Real-World Soundtracks](https://arxiv.org/abs/2110.09958) |  |

## Modules
| Module | Reference | Done |
| :---: | :---: | :---: |
| Depthwise-separable convolution | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357) | ✔ |
| Gated Linear Units (GLU) | [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) | ✔ |
| Sigmoid Linear Units (SiLU) | [Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning](https://arxiv.org/abs/1702.03118) | ✔ |
| Feature-wise Linear Modulation (FiLM) | [FiLM: Visual Reasoning with a General Conditioning Layer](https://arxiv.org/abs/1709.07871) | ✔ |
| Point-wise Convolutional Modulation (PoCM) | [LaSAFT: Latent Source Attentive Frequency Transformation for Conditioned Source Separation](https://arxiv.org/abs/2010.11631) | ✔ |

## Method related to training
| Method | Reference | Done |
| :---: | :---: | :---: |
| Pemutation invariant training (PIT) | [Multi-talker Speech Separation with Utterance-level Permutation Invariant Training of Deep Recurrent Neural Networks](https://arxiv.org/abs/1703.06284) | ✔ |
| One-and-rest PIT | [Recursive Speech Separation for Unknown Number of Speakers](https://arxiv.org/abs/1904.03065) | ✔ |
| Probabilistic PIT | [Probabilistic Permutation Invariant Training for Speech Separation](https://arxiv.org/abs/1908.01768) |  |
| Sinkhorn PIT | [Towards Listening to 10 People Simultaneously: An Efficient Permutation Invariant Training of Audio Source Separation Using Sinkhorn's Algorithm](https://arxiv.org/abs/2010.11871) | ✔ |
| Combination Loss | [All for One and One for All: Improving Music Separation by Bridging Networks](https://arxiv.org/abs/2010.04228) | ✔ |

## Example
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/train_librispeech.ipynb)

LibriSpeech example using [Conv-TasNet](https://arxiv.org/abs/1809.07454)

You can check other tutorials in `<REPOSITORY_ROOT>/egs/tutorials/`.

### 0. Preparation
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/common/
. ./prepare_librispeech.sh \
--librispeech_root <LIBRISPEECH_ROOT> \
--n_sources <#SPEAKERS>
```

### 1. Training
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./train.sh \
--exp_dir <OUTPUT_DIR>
```

If you want to resume training,
```sh
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. Evaluation
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./test.sh \
--exp_dir <OUTPUT_DIR>
```

### 3. Demo
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/conv-tasnet/
. ./demo.sh
```

## Pretrained Models
You need `gdown` to download pretrained models.
```sh
pip install gdown
```
You can load pretrained models.
```py
from models.conv_tasnet import ConvTasNet

model = ConvTasNet.build_from_pretrained(task="musdb18", sample_rate=44100, target="vocals")
```

See `PRETRAINED.md`, `egs/tutorials/hub/pretrained.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/hub/pretrained.ipynb) for details.

### Time Domain Wrappers for Time-Frequency Domain Models
See `egs/tutorials/hub/time-domain_wrapper.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/hub/time-domain_wrapper.ipynb).

### Speech Separation by Pretrained Models
See `egs/tutorials/hub/speech-separation.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/hub/speech-separation.ipynb).

### Music Source Separation by Pretrained Models
See `egs/tutorials/hub/music-source-separation.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/hub/music-source-separation.ipynb).

If you want to separate your own music file, see below:
- MMDenseLSTM: See `egs/tutorials/mm-dense-lstm/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/mm-dense-lstm/separate_music.ipynb).
- Conv-TasNet: See `egs/tutorials/conv-tasnet/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music.ipynb).
- UMX: See `egs/tutorials/umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_music.ipynb).
- X-UMX: See `egs/tutorials/x-umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/x-umx/separate_music.ipynb).
- D3Net: See `egs/tutorials/d3net/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate_music.ipynb).