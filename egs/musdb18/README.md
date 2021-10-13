# Music Source Separation using MUSDB18
## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDense | - | - | - | - | - | - |
| MMDenseLSTM | - | - | - | - | - | - |
| Conv-TasNet | 5.59 | 6.11 | 5.95 | 3.78 | 11.90 | 5.36 |
| CUNet | - | - | - | - | - | - |
| Meta-TasNet | - | - | - | - | - | - |
| UMX | - | - | - | - | - | - |
| X-UMX | - | - | - | - | - | - |
| D3Net | 6.63 | 6.40 | 5.24 | 4.58 | 13.24 | 5.71 |

## Separation Example
- Conv-TasNet: See `egs/tutorials/conv-tasnet/separate.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/conv-tasnet/separate.ipynb).
- UMX: See `egs/tutorials/umx/separate.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/umx/separate.ipynb).
- D3Net: See `egs/tutorials/d3net/separate.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate.ipynb).
