# DANetによる話者分離
データセット: LibriSpeech (NOT LibriMix)

## Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/danet/train_danet.ipynb)

Google Colaboratoryの環境で試すことができます．`train_danet.ipynb`を見てください．

## 実行方法
学習時と評価時でログが`train_<TIME_STAMP>.log`のように保存されます．
`<TIME_STAMP>`は`date "+%Y%m%d-%H%M%S"`で与えられ，これはタイムゾーンに依存しています．
そのため`export TZ=UTC-9`のように設定することを勧めます．
`TZ=UTC-9`は`Coordinated Universal Time +9 hours`を意味しています．

### 0. データセットの準備
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/common/
. ./prepare_librispeech.sh \
--librispeech_root <LIBRISPEECH_ROOT> \
--n_sources <#SPEAKERS>
```

### 1. 学習
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/danet/
. ./train.sh \
--exp_dir <OUTPUT_DIR>
```

学習を途中から再開したい場合，
```sh
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. 評価
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/danet/
. ./test.sh \
--exp_dir <OUTPUT_DIR>
```

### 3. デモ
```sh
cd <REPOSITORY_ROOT>/egs/tutorials/danet/
. ./demo.sh
```

## 結果
テストデータに対する評価．
