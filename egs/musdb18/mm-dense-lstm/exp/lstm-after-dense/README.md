# Example of outputs
I trained MMDenseLSTM and share the outputs. The networks are all trained by default setting.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV/view?usp=sharing](https://drive.google.com/file/d/1-2JGWMgVBdSj5zF9hl27jKhyX7GN-cOV/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1sjlU2QZPVyvBC4Ql8vOvuAAqBibTZQZj/view?usp=sharing](https://drive.google.com/file/d/1sjlU2QZPVyvBC4Ql8vOvuAAqBibTZQZj/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18 (training, test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| MMDenseLSTM (Sa) | 4.80 | 6.22 | 4.43 | 6.87 | 13.13 | 5.58 | Epoch is chosen by validation loss. |
| MMDenseLSTM (Sa) | 4.82 | 6.25 | 4.39 | 6.58 | 13.12 | 5.51 | After training. |