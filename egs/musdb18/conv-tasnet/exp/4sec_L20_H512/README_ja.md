# 出力結果の例
Conv-TasNetを学習させた結果を共有します．拡張に関しては`config/paper/augmentation.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1A6dIofHZJQCUkyq-vxZ6KbPmEHLcf4WK/view?usp=sharing](https://drive.google.com/file/d/1A6dIofHZJQCUkyq-vxZ6KbPmEHLcf4WK/view?usp=sharing)からダウンロードできます．これらは，`last.pth`と`best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1KDjKJsiVex0D2P1Q4q--6CXo514_x2xA/view?usp=sharing](https://drive.google.com/file/d/1KDjKJsiVex0D2P1Q4q--6CXo514_x2xA/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 5.32 | 6.06 | 4.00 | 6.04 | 12.33 | 5.35 | 検証ロスが最小となるエポックで学習を止めた場合． |
| Conv-TasNet | 44100 | 4 | 20 | 256 | 512 | 4.82 | 5.98 | 3.73 | 6.06 | 12.32 | 5.15 | 100エポック学習後 |