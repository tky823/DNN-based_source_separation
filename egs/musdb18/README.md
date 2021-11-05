# Music Source Separation using MUSDB18
## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Bass | Drums | Other | Vocals | Accompaniment | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDense | - | - | - | - | - | - |
| MMDenseLSTM | 4.80 | 6.22 | 4.43 | 6.87 | 13.13 | 5.58 |
| Conv-TasNet | 5.95 | 6.11 | 3.78 | 5.59 | 11.90 | 5.36 |
| CUNet | - | - | - | - | - | - |
| Meta-TasNet | - | - | - | - | - | - |
| UMX | 5.02 | 6.06 | 4.00 | 6.14 | 12.18 | 5.30 |
| X-UMX | 4.84 | 6.01 | 3.75 | 5.53 | 12.10 | 5.03 |
| D3Net | 5.24 | 6.71 | 4.59 | 6.97 | 13.22 | 5.88 |
| LaSAFT | - | - | - | - | - | - |

## Separation Example
- MMDenseLSTM: See `egs/tutorials/mm-dense-lstm/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/mm-dense-lstm/separate_music.ipynb).
- Conv-TasNet: See `egs/tutorials/conv-tasnet/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate_music.ipynb).
- UMX: See `egs/tutorials/umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate_music.ipynb).
- X-UMX: See `egs/tutorials/x-umx/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/x-umx/separate_music.ipynb).
- D3Net: See `egs/tutorials/d3net/separate_music.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate_music.ipynb).
