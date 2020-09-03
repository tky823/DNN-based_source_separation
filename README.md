# DNN-based source separation
A PyTorch implementation of DNN-based source separation.

## Model
| Model | Reference | Done | 
| :---: | :---: | :---: |
| WaveNet | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | ✔ |
| Wave-U-Net | [Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source Separation](https://arxiv.org/abs/1806.03185) |  |
| Deep clustering | [Single-Channel Multi-Speaker Separation using Deep Clustering](https://arxiv.org/abs/1607.02173) |  |
| Chimera++ | [Alternative Objective Functions for Deep Clustering](https://www.merl.com/publications/docs/TR2018-005.pdf) |  |
| DANet | [Deep attractor network for single-microphone speaker separation](https://arxiv.org/abs/1611.08930) |  |
| ADANet | [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634) |  |
| TasNet | [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) | ✔ |
| Conv-TasNet | [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) | ✔ |
| DPRNN-TasNet | [Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation](https://arxiv.org/abs/1910.06379) | ✔ |
| Gated DPRNN-TasNet | [Voice Separation with an Unknown Number of Multiple Speakers](https://arxiv.org/abs/2002.08933) |  |
| DeepCASA | [Divide and conquer: A deep
casa approach to talker-independent monaural speaker separation](https://arxiv.org/abs/1904.11148) |  |
| FurcaNet | [FurcaNet: An end-to-end deep gated convolutional, long short-term memory, deep neural networks for single channel speech separation](https://arxiv.org/abs/1902.00651) |  |
| Wavesplit | [Wavesplit: End-to-End Speech Separation by Speaker Clustering](https://arxiv.org/abs/2002.08933) |  |

## Example
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
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./train.sh <OUTPUT_DIR>
```

If you want to resume training,
```
. ./train.sh <OUTPUT_DIR> <MODEL_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./test.sh <OUTPUT_DIR>
```

### 3. Demo
```
cd <REPOSITORY_ROOT>/egs/librispeech/conv_tasnet/
. ./demo.sh
```

## Version
v0.0.0: Initial version. LibriSpeech Conv-TasNet & DPRNN-TasNet examples are included.

v0.0.1: Dataset is renamed.

v0.1.0: Dataset structure is changed.

v0.1.1: DANet is included.
