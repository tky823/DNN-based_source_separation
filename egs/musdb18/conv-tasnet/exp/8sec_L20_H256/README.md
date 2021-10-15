# Example of outputs
I trained Conv-TasNet and share the outputs. See `8sec_L20_H256/config/augmentation-*.yaml` for augmentation.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1pA-jt_709cq8Pl6TAix3Yf2ei6UNPldn/view?usp=sharing](https://drive.google.com/file/d/1pA-jt_709cq8Pl6TAix3Yf2ei6UNPldn/view?usp=sharing) that includes `last.pth` and `best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing](https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.59 | 6.11 | 5.95 | 3.78 | 11.90 | 5.36 | Epoch is chosen by validation loss. |