# 出力結果の例
D3Netを学習させた結果を共有します．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing](https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing)からダウンロードできます．これらは，`bass/last.pth`，`bass/best.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[]()からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Vocals | Bass | Drums | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | - | - | - | - | - | - | 検証ロスが最小となるエポックで学習を止めた場合 |
| D3Net | 44100 | - | - | - | - | - | - | 100エポック学習後 |