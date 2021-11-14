# Example of outputs
I trained UMX and share the outputs. The networks are all trained by default setting.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1sqlK26fLJ6ns-NOxCrxhwI92wv45QPCB/view?usp=sharing](https://drive.google.com/file/d/1sqlK26fLJ6ns-NOxCrxhwI92wv45QPCB/view?usp=sharing), which includes `drums/last.pth`, `drums/best.pth`, `bass/last.pth`, ..., `vocals/best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1oGinge0ScazijYk7e5lFepoBUECtrZoj/view?usp=sharing](https://drive.google.com/file/d/1oGinge0ScazijYk7e5lFepoBUECtrZoj/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)
- Dataset: MUSDB18 (training, test)

| Model | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UMX | 5.02 | 6.06 | 4.00 | 6.14 | 12.18 | 5.30 | Epoch is chosen by validation loss. |
| UMX | 5.00 | 6.15 | 4.04 | 5.75 | 12.35 | 5.23 | After training. |