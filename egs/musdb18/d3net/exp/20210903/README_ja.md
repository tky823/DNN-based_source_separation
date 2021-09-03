# Example of outputs
D3Netを学習させた結果を共有します．ネットワークは全てデフォルトの設定で学習済み．
- `json.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1hdgtTvLmHXDbYRxAaGuBJATtG9hcVGOK/view?usp=sharing](https://drive.google.com/file/d/1hdgtTvLmHXDbYRxAaGuBJATtG9hcVGOK/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 学習済みモデルを使って分離を試すことができます．`egs/tutorials/d3net/separate.ipynb`を見るか， [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tky823/DNN-based_source_separation/blob/main/egs/tutorials/d3net/separate.ipynb)にとんでください．

## Results
SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - | - |
| D3Net | 7.02 | 6.58 | 4.88 | 4.77 | 13.38 | 5.81 | Epoch is chosen by validation loss. |
| D3Net | 7.08 | 6.54 | 4.93 | 4.72 | 13.41 | 5.82 | After 50 epochs. |
