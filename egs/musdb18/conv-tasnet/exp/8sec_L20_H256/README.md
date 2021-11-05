# Example of outputs
I trained Conv-TasNet and share the outputs. See `config/paper/augmentation.yaml` for augmentation.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ/view?usp=sharing](https://drive.google.com/file/d/1C4uv2z0w1s4rudIMaErLyEccNprJQWSZ/view?usp=sharing) that includes `last.pth` and `best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing](https://drive.google.com/file/d/1aH3n6obNBP2j1vjaz-YfUtl_U2YJ6Mu7/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Bass | Drums | Other | Vocals | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.95 | 6.11 | 3.78 | 5.59 | 11.90 | 5.36 | Epoch is chosen by validation loss. |
| Conv-TasNet | 44100 | 8 | 20 | 256 | 256 | 5.13 | 6.10 | 3.57 | 5.82 | 12.00 | 5.16 | After 100 epochs. |