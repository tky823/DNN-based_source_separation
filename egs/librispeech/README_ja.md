# LibriSpeech
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/master/egs/librispeech/conv-tasnet/train_conv-tasnet.ipynb)

## 実行方法
学習や評価時には，`train_<TIME_STAMP>.log`のようにログが保存される．
`<TIME_STAMP>`は`date "+%Y%m%d-%H%M%S"`で生成され，タイムゾーンに依存する．
そのため，`TZ=UTC-9 date "+%Y%m%d-%H%M%S"`のようにタイムゾーンを指定することを勧める．
例えば，`TZ=UTC-9`は世界標準時+9時間を意味する（日本のタイムゾーン）．

### 0. データセットの準備
```
cd <REPOSITORY_ROOT>/egs/librispeech/common/
. ./prepare.sh <DATASET_DIR> <#SPEAKERS>
```

### 1. 学習
```
cd <REPOSITORY_ROOT>/egs/librispeech/<MODEL_NAME>/
. ./train.sh <OUTPUT_DIR>
```

学習を途中から再開したい場合，
```
. ./train.sh <OUTPUT_DIR> <MODEL_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/librispeech/<MODEL_NAME>/
. ./test.sh <OUTPUT_DIR>
```

### 3. デモンストレーション
```
cd <REPOSITORY_ROOT>/egs/librispeech/<MODEL_NAME>/
. ./demo.sh
```

## 結果
テストデータによる評価
学習済みモデルはサブディレクトリに保存されている．これらは全てGoogle Colaboratory上で学習されたもの．
ただし，時間制約の都合上，ネットワーク構造は元の論文と変えている可能性があるため注意すること．

| モデル | SI-SDRi [dB] | PESQ |
| :---: | :---: | :---: |
| DANet |  |  |
| ADANet |  |  |
| TasNet |  |  |
| Conv-TasNet |  |  |
| DPRNN-TasNet |  |  |
