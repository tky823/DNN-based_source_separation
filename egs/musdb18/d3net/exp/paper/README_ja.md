# 出力結果の例
D3Netを学習させた結果を共有します．ネットワークはランダムスケールを除いて，デフォルトの設定で学習済み．`config/improved/augmentation-*.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1D1Q097xz9aaiFUPtbLOjAgVYPPLnOsiA/view?usp=sharing](https://drive.google.com/file/d/1D1Q097xz9aaiFUPtbLOjAgVYPPLnOsiA/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1GU2CBgHKrRcFJZHzwckcdHAgVA7xqRGx/view?usp=sharing](https://drive.google.com/file/d/1GU2CBgHKrRcFJZHzwckcdHAgVA7xqRGx/view?usp=sharing)からダウンロードできます．

## 実験結果
SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net w/o dilation | - | - | - | - | - | - | - |
| D3Net standard dilation | - | - | - | - | - | - | - |
| D3Net | 6.98 | 6.55 | 4.99 | 4.81 | 13.45 | 5.83 | 検証ロスが最小となるエポックで学習を止めた場合． |
| D3Net | 6.99 | 6.58 | 5.07 | 4.66 | 13.38 | 5.83 | 50エポック後． |