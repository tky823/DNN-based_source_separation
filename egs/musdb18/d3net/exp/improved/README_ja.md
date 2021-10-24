# 出力結果の例
D3Netを学習させた結果を共有します．
- `config.zip`，`log.zip`，`loss.zip`を解凍する必要があります．
- 学習済みモデルは[https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing](https://drive.google.com/file/d/1pce_DYaeDYMvsKHmDAvL1Cww_1I3pnhr/view?usp=sharing)からダウンロードできます．これらは，`bass/last.pth`，`bass/best.pth`，...，`vocals/best.pth`を含んでいます．
- 出力されたJSONファイルは[https://drive.google.com/file/d/1OOKBfY0c08LodjKeWXA4ETGWMwzCSbdN/view?usp=sharing](https://drive.google.com/file/d/1OOKBfY0c08LodjKeWXA4ETGWMwzCSbdN/view?usp=sharing)からダウンロードできます．

## 実験結果
- SDR [dB] (`museval`によって計算された各曲のSDRの中央値の中央値)

| Model | Sampling rate [Hz] | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D3Net | 44100 | 4.98 | 6.40 | 4.40 | 7.06 | 13.42 | 5.71 | 検証ロスが最小となるエポックで学習を止めた場合 |
| D3Net | 44100 | 5.02 | 6.63 | 4.43 | 7.03 | 13.42 | 5.78 | 100エポック学習後 |