# 出力結果の例
UMXを学習させた結果を共有します．ネットワークはデフォルトの設定で学習済み．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/18pj2ubYnZPSQWPpHaREAcbmrNzEihNHO/view?usp=sharing](https://drive.google.com/file/d/18pj2ubYnZPSQWPpHaREAcbmrNzEihNHO/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1YoHMB79oSmDUdcE1OM06gNkFJVkkP4VF/view?usp=sharing](https://drive.google.com/file/d/1YoHMB79oSmDUdcE1OM06gNkFJVkkP4VF/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
- データセット: MUSDB18HQ (学習)，MUSDB18（テスト）

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX-HQ | 4.85 | 5.94 | 4.01 | 6.08 | 12.05 | 5.22 | 検証ロスが最小となるエポックで学習を止めた場合 |
| UMX-HQ | 4.90 | 6.12 | 3.99 | 5.92 | 12.19 | 5.23 | 学習後 |