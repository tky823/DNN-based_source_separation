# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/18pj2ubYnZPSQWPpHaREAcbmrNzEihNHO/view?usp=sharing](https://drive.google.com/file/d/18pj2ubYnZPSQWPpHaREAcbmrNzEihNHO/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1YoHMB79oSmDUdcE1OM06gNkFJVkkP4VF/view?usp=sharing](https://drive.google.com/file/d/1YoHMB79oSmDUdcE1OM06gNkFJVkkP4VF/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18HQ (training), MUSDB18 (test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX-HQ | 4.85 | 5.94 | 4.01 | 6.08 | 12.05 | 5.22 | Epoch is chosen by validation loss. |
| UMX-HQ | 4.90 | 6.12 | 3.99 | 5.92 | 12.19 | 5.23 | After training. |