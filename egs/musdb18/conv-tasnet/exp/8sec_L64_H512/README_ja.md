# 出力結果の例
Conv-TasNetを学習させた結果を共有します．拡張に関しては`8sec_L64_H512/config/augmentation-*.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1yQTdGu2jigAHotJJ7JAQcfqedACCP40t/view?usp=sharing](https://drive.google.com/file/d/1yQTdGu2jigAHotJJ7JAQcfqedACCP40t/view?usp=sharing)からダウンロードできます．これらは，`last.pth`と`best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1R-63xiocr6Cjp84O9pKej-X2iZFb-QOl/view?usp=sharing](https://drive.google.com/file/d/1R-63xiocr6Cjp84O9pKej-X2iZFb-QOl/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 64 | 256 | 512 | 5.91 | 5.82 | 5.38 | 3.51 | 11.85 | 5.16 | 検証ロスが最小となるエポックで学習を止めた場合． |
| Conv-TasNet | 44100 | 8 | 64 | 256 | 512 | 6.02 | 5.79 | 5.33 | 3.48 | 11.91 | 5.16 | 100エポック学習後． |