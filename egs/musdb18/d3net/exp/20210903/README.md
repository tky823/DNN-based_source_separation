# Example of outputs
I trained D3Net and share the outputs. The networks are all trained by default setting.
- You have to unzip `json.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1hdgtTvLmHXDbYRxAaGuBJATtG9hcVGOK/view?usp=sharing](https://drive.google.com/file/d/1hdgtTvLmHXDbYRxAaGuBJATtG9hcVGOK/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can separate your audio using these pretrained models. See `egs/tutorials/d3net/separate.ipynb` or click [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate.ipynb).

## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - | - |
| D3Net | 7.02 | 6.58 | 4.88 | 4.77 | 13.38 | 5.81 | Epoch is chosen by validation loss. |
| D3Net | 7.08 | 6.54 | 4.93 | 4.72 | 13.41 | 5.82 | After 50 epochs. |
