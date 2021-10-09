# Conv-TasNet
## How to Run
### 0. Preparation
Build environment by pip
```
cd <REPOSITORY_ROOT>/egs/musdb18/
pip install -r requirements.txt
```
or by conda.
```
cd <REPOSITORY_ROOT>/egs/musdb18/
conda env create -f environment-gpu.yaml
```

Download MUSDB18 dataset and convert to `.wav`.
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18_root <MUSDB18_ROOT> \
--is_hq 0 \
--to_wav 1
```
If you want to download MUSDB18-HQ dataset, 
```
cd <REPOSITORY_ROOT>/egs/musdb18/common/
. ./prepare_musdb18.sh \
--musdb18_root <MUSDB18HQ_ROOT> \
--is_hq 1
```

### 1. Training
```
cd <REPOSITORY_ROOT>/egs/musdb18/conv-tasnet/
. ./train.sh \
--exp_dir <OUTPUT_DIR>
```

If you want to resume training,
```
. ./train.sh \
--exp_dir <OUTPUT_DIR> \
--continue_from <MODEL_PATH>
```

### 2. Evaluation
```
cd <REPOSITORY_ROOT>/egs/musdb18/conv-tasnet/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | - | - | - | - | - | - | - |

- You can separate your audio using these pretrained models. See `egs/tutorials/conv-tasnet/separate.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate.ipynb).