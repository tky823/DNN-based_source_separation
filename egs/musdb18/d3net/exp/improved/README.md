# Example of outputs
I trained D3Net and share the outputs. The networks are all trained by default setting except for random scaling. See `config/improved/augmentation-*.yaml`.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1ZDWDO56m4oDg_R6xbV3lCZnpQeseXGXw/view?usp=sharing](https://drive.google.com/file/d/1ZDWDO56m4oDg_R6xbV3lCZnpQeseXGXw/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/14fadm_Y2y4rh6pw23sWkLM28LWys5ukw/view?usp=sharing](https://drive.google.com/file/d/14fadm_Y2y4rh6pw23sWkLM28LWys5ukw/view?usp=sharing).
- You can separate your audio using these pretrained models. See `egs/tutorials/d3net/separate.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate.ipynb).

## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - | - |
| D3Net | 6.98 | 6.55 | 4.99 | 4.81 | 13.45 | 5.83 | Epoch is chosen by validation loss. |
| D3Net | 6.99 | 6.58 | 5.07 | 4.66 | 13.38 | 5.83 | After 50 epochs. |