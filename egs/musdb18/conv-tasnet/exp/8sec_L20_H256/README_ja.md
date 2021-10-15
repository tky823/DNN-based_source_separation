# 出力結果の例
Conv-TasNetを学習させた結果を共有します．拡張に関しては`8sec_L20_H256/config/augmentation-*.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1pA-jt_709cq8Pl6TAix3Yf2ei6UNPldn/view?usp=sharing](https://drive.google.com/file/d/1pA-jt_709cq8Pl6TAix3Yf2ei6UNPldn/view?usp=sharing)からダウンロードできます．これらは，`last.pth`と`best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing](https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.59 | 6.11 | 5.95 | 3.78 | 11.90 | 5.36 | 検証ロスが最小となるエポックで学習を止めた場合． |