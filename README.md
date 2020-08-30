# DNN-based source separation
A PyTorch implementation of DNN-based source separation.

## Model
| Model | Reference |
| :---: | :---: |
| WaveNet | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) |
| DANet | [Deep attractor network for single-microphone speaker separation](https://arxiv.org/abs/1611.08930) |
| ADANet | [Speaker-independent Speech Separation with Deep Attractor Network](https://arxiv.org/abs/1707.03634) |
| TasNet | [TasNet: time-domain audio separation network for real-time, single-channel speech separation](https://arxiv.org/abs/1711.00541) |
| Conv-TasNet | [Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation](https://arxiv.org/abs/1809.07454) |
| DPRNN-TasNet | [Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation](https://arxiv.org/abs/1910.06379) |

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
