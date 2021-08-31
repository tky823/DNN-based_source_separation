# D3Net
Reference: [D3Net: Densely connected multidilated DenseNet for music source separation](https://arxiv.org/abs/2010.01733)

## How to Run
### 0. Preparation
Build environment by pip
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
pip install -r requirements.txt
```
or by conda.
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
conda env create -f environment-gpu.yaml
```

Download MUSDB18 dataset and convert to `.wav`.
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18_root <MUSDB18_ROOT> \
--to_wav 1
```
If you want to download MUSDB18-HQ dataset, 
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18hq_root <MUSDB18HQ_ROOT> \
--is_hq 1
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--target <TARGET> \
--config_path <CONFIG_PATH>
```

If you want to resume training,
```
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH> \
--target <TARGET> \
--config_path <CONFIG_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/musdb18/d3net/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - |
| D3Net | 6.82 | 6.29 | 4.77 | 4.51 | 13.06 | 5.60 |