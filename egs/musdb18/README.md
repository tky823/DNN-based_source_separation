# Music Source Separation using MUSDB18
## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Vocals | Bass | Drums | Other | Accompaniment | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDense | - | - | - | - | - | - |
| MMDenseLSTM | - | - | - | - | - | - |
| Conv-TasNet | 5.59 | 5.95 | 6.11 | 3.78 | 11.90 | 5.36 |
| CUNet | - | - | - | - | - | - |
| Meta-TasNet | - | - | - | - | - | - |
| UMX | 5.71 | 6.00 | 3.99 | 4.82 | 12.14 | 5.13 |
| X-UMX | - | - | - | - | - | - |
| D3Net | 6.63 | 5.24 | 6.40 | 4.58 | 13.24 | 5.71 |

## Separation Example
- Conv-TasNet: See `egs/tutorials/conv-tasnet/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music.ipynb).
- UMX: See `egs/tutorials/umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_music.ipynb).
- D3Net: See `egs/tutorials/d3net/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate_music.ipynb).
