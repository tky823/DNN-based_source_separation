# LibriSpeechデータセットによる音源分離のチュートリアル
学習や評価時には，`train_<TIME_STAMP>.log`のようにログが保存される．
`<TIME_STAMP>`は`date "+%Y%m%d-%H%M%S"`で生成され，タイムゾーンに依存する．
そのため，`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`のようにタイムゾーンを指定することを勧める．
例えば，`TZ=UTC-9`は世界標準時+9時間を意味する（日本のタイムゾーン）．

2話者混合のテストデータによる評価．
学習済みモデルはサブディレクトリに保存されている．これらは全てGoogle Colaboratory上で学習されたもの．
ただし，時間制約の都合上，ネットワーク構造は元の論文と変えている可能性があるため注意すること．

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

# スピーチデータセットによる距離学習のチュートリアル
## triplet-loss
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/triplet-loss/speech-commands_ja.ipynb)

## siamese-net
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/siamese-net/speech-commands_ja.ipynb)

# torchaudio.datasetsのためのチュートリアル
## cmu-arctic
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/cmu-arctic/cmu-arctic_ja.ipynb)

## yes-no
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/yes-no/yes-no_ja.ipynb)