# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting except for random scaling. See `config/paper/augmentation.yaml`.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [](), which includes `last.pth` and `best.pth`.
- You can download output JSON files from []().

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18 (training, test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| X-UMX | 4.84 | 6.01 | 3.75 | 5.53 | 12.10 | 5.03 | Epoch is chosen by validation loss. |
| X-UMX | 4.47 | 5.77 | 3.53 | 5.53 | 11.93 | 4.82 | After training. |