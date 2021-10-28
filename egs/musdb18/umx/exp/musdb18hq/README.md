# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting except for random scaling. See `config/paper/augmentation.yaml`.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [](), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from []().

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18HQ (training, test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX-HQ | - | - | - | - | - | - | Epoch is chosen by validation loss. |
| UMX-HQ | - | - | - | - | - | - | After 100 epochs. |