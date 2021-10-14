# 出力結果の例
UMXを学習させた結果を共有します．ネットワークはランダムスケールを除いて，デフォルトの設定で学習済み．`config/improved/augmentation-*.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T/view?usp=sharing](https://drive.google.com/file/d/1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1_Do6fy2fxN86EEy2_6qBolloJ-D-VXyO/view?usp=sharing](https://drive.google.com/file/d/1_Do6fy2fxN86EEy2_6qBolloJ-D-VXyO/view?usp=sharing)からダウンロードできます．

## 実験結果
SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
| Model | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 5.71 | 6.00 | 4.82 | 3.99 | 12.14 | 5.13 | 検証ロスが最小となるエポックで学習を止めた場合． |
| UMX | 5.81 | 6.09 | 4.69 | 3.66 | 12.07 | 5.06 | 100エポック後． |