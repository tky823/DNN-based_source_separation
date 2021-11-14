# 出力結果の例
UMXを学習させた結果を共有します．ネットワークはデフォルトの設定で学習済み．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1W0fNeGoqQU6Zj0KHA8n3n6iSonuiaHdQ/view?usp=sharing](https://drive.google.com/file/d/1W0fNeGoqQU6Zj0KHA8n3n6iSonuiaHdQ/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1bAtYVpo0nIkDPMJgCixyhuwdhkOUHI2j/view?usp=sharing](https://drive.google.com/file/d/1bAtYVpo0nIkDPMJgCixyhuwdhkOUHI2j/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
- データセット: MUSDB18HQ (学習・テスト)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 4.92 | 5.89 | 3.82 | 5.77 | 12.18 | 5.09 | 検証ロスが最小となるエポックで学習を止めた場合 |
| UMX | 4.84 | 5.77 | 3.87 | 5.71 | 12.17 | 5.05 | 100エポック後 |