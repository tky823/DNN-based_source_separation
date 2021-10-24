# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting except for random scaling. See `config/paper/augmentation.yaml`.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T/view?usp=sharing](https://drive.google.com/file/d/1C67tgD79YIe-uEs31NTPMxuh7JNLPB7T/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1_Do6fy2fxN86EEy2_6qBolloJ-D-VXyO/view?usp=sharing](https://drive.google.com/file/d/1_Do6fy2fxN86EEy2_6qBolloJ-D-VXyO/view?usp=sharing).

## Results
SDR [dB] (median of median SDR of each song computed by `museval`)
| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 4.82 | 6.00 | 3.99 | 5.71 | 12.14 | 5.13 | Epoch is chosen by validation loss. |
| UMX | 4.69 | 6.09 | 3.66 | 5.81 | 12.07 | 5.06 | After 100 epochs. |