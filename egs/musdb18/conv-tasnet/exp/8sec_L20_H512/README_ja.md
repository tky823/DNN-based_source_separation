# 出力結果の例
Conv-TasNetを学習させた結果を共有します．拡張に関しては`4sec_L20_H512/config/augmentation-*.yaml`を見てください．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1a-IQn3hsN84N2X_WF84uXbwe7VAk_vaB/view?usp=sharing](https://drive.google.com/file/d/1a-IQn3hsN84N2X_WF84uXbwe7VAk_vaB/view?usp=sharing)からダウンロードできます．これらは，`last.pth`と`best.pth`を含んでいます．
- 出力されたJSONファイルは[]()からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 512 | - | - | - | - | - | - | - |