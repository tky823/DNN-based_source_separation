# D3Net
Reference: [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733)

## How to Run
### 0. Preparation
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh --musdb18_root <MUSDB18_ROOT>
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./train.sh --exp_dir <OUTPUT_DIR> --target <TARGET> --config_path <CONFIG_PATH>
```

If you want to resume training,
```
. ./train.sh --exp_dir <OUTPUT_DIR> --continue_from <MODEL_PATH> --target <TARGET> --config_path <CONFIG_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## Results
WIP