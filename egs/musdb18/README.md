# Music Source Separation using MUSDB18
## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Bass | Drums | Other | Vocals | Accompaniment | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDense | - | - | - | - | - | - |
| MMDenseLSTM | - | - | - | - | - | - |
| Conv-TasNet | 5.95 | 6.11 | 3.78 | 5.59 | 11.90 | 5.36 |
| CUNet | - | - | - | - | - | - |
| Meta-TasNet | - | - | - | - | - | - |
| UMX | 6.00 | 3.99 | 4.82 | 5.71 | 12.14 | 5.13 |
| X-UMX | - | - | - | - | - | - |
| D3Net | 5.24 | 6.71 | 4.59 | 6.97 | 13.22 | 5.88 |

## Separation Example
- Conv-TasNet: See `egs/tutorials/conv-tasnet/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music.ipynb).
- UMX: See `egs/tutorials/umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_music.ipynb).
- D3Net: See `egs/tutorials/d3net/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate_music.ipynb).
