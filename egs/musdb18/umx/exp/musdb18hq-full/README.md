# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1W0fNeGoqQU6Zj0KHA8n3n6iSonuiaHdQ/view?usp=sharing](https://drive.google.com/file/d/1W0fNeGoqQU6Zj0KHA8n3n6iSonuiaHdQ/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1bAtYVpo0nIkDPMJgCixyhuwdhkOUHI2j/view?usp=sharing](https://drive.google.com/file/d/1bAtYVpo0nIkDPMJgCixyhuwdhkOUHI2j/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18HQ (training, test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 4.92 | 5.89 | 3.82 | 5.77 | 12.18 | 5.09 | Epoch is chosen by validation loss. |
| UMX | 4.84 | 5.77 | 3.87 | 5.71 | 12.17 | 5.05 | After 100 epochs. |