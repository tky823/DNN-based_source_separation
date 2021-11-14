# Open-Unmix (UMX)
Reference: [Open-Unmix - A Reference Implementation for Music Source Separation](https://hal.inria.fr/hal-02293689/document)

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
cd <REPOSITORY_ROOT>/egs/musdb18/umx/
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
cd <REPOSITORY_ROOT>/egs/musdb18/umx/
. ./test.sh --exp_dir <OUTPUT_DIR>
```

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 5.02 | 6.06 | 4.00 | 6.14 | 12.18 | 5.30 | Epoch is chosen by validation loss. |
| UMX | 5.00 | 6.15 | 4.04 | 5.75 | 12.35 | 5.23 | After training. |
| UMX-HQ | 4.85 | 5.94 | 4.01 | 6.08 | 12.05 | 5.22 | Epoch is chosen by validation loss. |
| UMX-HQ | 4.90 | 6.12 | 3.99 | 5.92 | 12.19 | 5.23 | After training. |
| UMX | 5.23 | 5.73 | 4.02 | 6.32 | - | 5.33 | Official report. |
| UMX-HQ | - | - | - | - | - | - | Official report. |

- You can separate your audio using these pretrained models. See `egs/tutorials/umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_music.ipynb).