# 出力結果の例
D3Netを学習させた結果を共有します．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1B4e4e-8-T1oKzSg8WJ8RIbZ99QASamPB/view?usp=sharing](https://drive.google.com/file/d/1B4e4e-8-T1oKzSg8WJ8RIbZ99QASamPB/view?usp=sharing)からダウンロードできます．これらは，`bass/last.pth`，`bass/best.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1BYhKXMdnJL7RD71Gy9DWjK-u1XU08CKZ/view?usp=sharing](https://drive.google.com/file/d/1BYhKXMdnJL7RD71Gy9DWjK-u1XU08CKZ/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | 5.24 | 6.71 | 4.59 | 6.97 | 13.22 | 5.88 | 検証ロスが最小となるエポックで学習を止めた場合 |
| D3Net | 44100 | 5.24 | 6.65 | 4.52 | 6.91 | 13.29 | 5.83 | 100エポック学習後 |