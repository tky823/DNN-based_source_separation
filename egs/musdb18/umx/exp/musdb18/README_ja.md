# 出力結果の例
UMXを学習させた結果を共有します．ネットワークはデフォルトの設定で学習済み．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1sqlK26fLJ6ns-NOxCrxhwI92wv45QPCB/view?usp=sharing](https://drive.google.com/file/d/1sqlK26fLJ6ns-NOxCrxhwI92wv45QPCB/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1oGinge0ScazijYk7e5lFepoBUECtrZoj/view?usp=sharing](https://drive.google.com/file/d/1oGinge0ScazijYk7e5lFepoBUECtrZoj/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
- データセット: MUSDB18 (学習・テスト)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 5.02 | 6.06 | 4.00 | 6.14 | 12.18 | 5.30 | 検証ロスが最小となるエポックで学習を止めた場合 |
| UMX | 5.00 | 6.15 | 4.04 | 5.75 | 12.35 | 5.23 | 学習後 |