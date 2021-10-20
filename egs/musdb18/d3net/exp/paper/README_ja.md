# 出力結果の例
D3Netを学習させた結果を共有します．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1We9ea5qe3Hhcw28w1XZl2KKogW9wdzKF/view?usp=sharing](https://drive.google.com/file/d/1We9ea5qe3Hhcw28w1XZl2KKogW9wdzKF/view?usp=sharing)からダウンロードできます．これらは，`bass/last.pth`，`bass/best.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1hrhUekc-BJbVeZw6xdTPU2Gg3Dael5vM/view?usp=sharing](https://drive.google.com/file/d/1hrhUekc-BJbVeZw6xdTPU2Gg3Dael5vM/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Vocals | Bass | Drums | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | 6.88 | 6.34 | 5.26 | 4.54 | 13.35 | 5.76 | 検証ロスが最小となるエポックで学習を止めた場合 |
| D3Net | 44100 | 6.89 | 6.38 | 5.27 | 4.57 | 13.34 | 5.77 | 100エポック学習後 |