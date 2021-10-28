# 出力結果の例
UMXを学習させた結果を共有します．ネットワークはランダムスケールを除いて，デフォルトの設定で学習済み．`config/paper/augmentation.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[]()からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[]()からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
- データセット: MUSDB18HQ (学習・テスト)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX-HQ | - | - | - | - | - | - | 検証ロスが最小となるエポックで学習を止めた場合 |
| UMX-HQ | - | - | - | - | - | - | 100エポック後 |