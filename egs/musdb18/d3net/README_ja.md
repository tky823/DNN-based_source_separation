# D3Net
Reference: [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733)

## 実行方法
### 0. データセットの準備
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh --musdb18_root <MUSDB18_ROOT>
```

### 1. 学習
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./train.sh --exp_dir <OUTPUT_DIR> --target <TARGET> --config_path <CONFIG_PATH>
```

学習を途中から再開したい場合，
```
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH> --target <TARGET> --config_path <CONFIG_PATH>
```

### 2. 評価
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## 実験結果
WIP