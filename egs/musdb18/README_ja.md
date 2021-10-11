# MUSDB18を使用した楽音分離
## 実験結果
SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 5.59 | 6.11 | 5.95 | 3.78 | 11.90 | 5.36 |
| CUNet | - | - | - | - | - | - |
| Meta-TasNet | - | - | - | - | - | - |
| UMX | - | - | - | - | - | - |
| X-UMX | - | - | - | - | - | - |
| D3Net | 6.63 | 6.40 | 5.24 | 4.58 | 13.24 | 5.71 |

## 分離の例
- Conv-TasNet: `egs/tutorials/conv-tasnet/separate_ja.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_ja.ipynb)にとんでください．
- D3Net: `egs/tutorials/d3net/separate_ja.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate_ja.ipynb)にとんでください．
