# 出力結果の例
MMDenseLSTMを学習させた結果を共有します．ネットワークはデフォルトの設定で学習済み
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV/view?usp=sharing](https://drive.google.com/file/d/1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV/view?usp=sharing)からダウンロードできます．これらは，`drums/last.pth`，`drums/best.pth`，`bass/last.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1sjlU2QZPVyvBC4Ql8vOvuAAqBibTZQZj/view?usp=sharing](https://drive.google.com/file/d/1sjlU2QZPVyvBC4Ql8vOvuAAqBibTZQZj/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)
- データセット: MUSDB18 (学習・テスト)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDenseLSTM (Sa) | 4.80 | 6.22 | 4.43 | 6.87 | 13.13 | 5.58 | 検証ロスが最小となるエポックで学習を止めた場合 |
| MMDenseLSTM (Sa) | 4.82 | 6.25 | 4.39 | 6.58 | 13.12 | 5.51 | 学習後 |