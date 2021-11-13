# 出力結果の例
UMXを学習させた結果を共有します．ネットワークはデフォルトの設定で学習済み．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1yQC00DFvHgs4U012Wzcg69lvRxw5K9Jj/view?usp=sharing](https://drive.google.com/file/d/1yQC00DFvHgs4U012Wzcg69lvRxw5K9Jj/view?usp=sharing)からダウンロードできます．これらは，`last.pth`，`best.pth`を含んでいます．
- 出力されたJSONファイルは[]()からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
- データセット: MUSDB18 (学習・テスト)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-UMX | 4.84 | 6.01 | 3.75 | 5.53 | 12.10 | 5.03 | 検証ロスが最小となるエポックで学習を止めた場合 |
| X-UMX | 4.47 | 5.77 | 3.53 | 5.53 | 11.93 | 4.82 | 学習後 |