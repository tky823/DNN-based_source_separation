# Tutorials for source separation using LibriSpeech dataset
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time zone.
I recommend that you specify your time zone like `TZ=UTC-9 date "+%Y%m%d-%H%M%S"`.
Here, `TZ=UTC-9` means `Coordinated Universal Time +9 hours`.

Evaluation for test data of 2 speakers.
Models are placed in sub-directory. These models are trained on Google Colaboratory.
Network configuration may be different from original papers.

| Model | SI-SDRi [dB] | SDRi [dB] | PESQ |
| :---: | :---: | :---: | :---: |
| DANet |  |  |  |
| ADANet |  |  |  |
| TasNet |  |  |  |
| Conv-TasNet |  |  |  |
| DPRNN-TasNet |  |  |  |

## danet
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/danet/train_danet.ipynb)

## adanet
Coming soon...

## conv-tasnet
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/train_conv-tasnet.ipynb)

## dprnn-tasnet
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/dprnn-tasnet/train_dprnn-tasnet.ipynb)

## orpit_conv-tasnet
Coming soon...

# Tutorials for metric learning using speech dataset
## triplet-loss
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/triplet-loss/speech-commands.ipynb)

## siamese-net
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/siamese-net/speech-commands.ipynb)

# Tutorials for torchaudio.datasets
## cmu-arctic
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/cmu-arctic/cmu-arctic.ipynb)

## yes-no
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/yes-no/yes-no.ipynb)