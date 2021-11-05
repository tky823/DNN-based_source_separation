# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting except for random scaling. See `config/paper/augmentation.yaml`.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [](), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from []().

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18 (training, test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDenseLSTM (Sa) | 4.80 | 6.22 | 4.43 | 6.87 | 13.13 | 5.58 | Epoch is chosen by validation loss. |
| MMDenseLSTM (Sa) | 4.82 | 6.25 | 4.39 | 6.58 | 13.12 | 5.51 | After training. |