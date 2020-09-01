# LibriSpeech

## 実行方法
At the training and evaluation stage, log is saved like `train_<TIME_STAMP>.log`.
`<TIME_STAMP>` is given by `date "+%Y%m%d-%H%M%S"`, and it depends on time.
I recommend specify your time zone like `TZ=UTC-9 date "+%Y%m%d-%H%M%S"`.
Here, `TZ=UTC-9` means `Coordinated Universal Time +9 hours`.

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

## Results
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
