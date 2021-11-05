# 出力結果の例
Conv-TasNetを学習させた結果を共有します．拡張に関しては`config/paper/augmentation.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ/view?usp=sharing](https://drive.google.com/file/d/1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ/view?usp=sharing)からダウンロードできます．これらは，`last.pth`と`best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing](https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.95 | 6.11 | 3.78 | 5.59 | 11.90 | 5.36 | 検証ロスが最小となるエポックで学習を止めた場合 |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.13 | 6.10 | 3.57 | 5.82 | 12.00 | 5.16 | 100エポック学習後 |