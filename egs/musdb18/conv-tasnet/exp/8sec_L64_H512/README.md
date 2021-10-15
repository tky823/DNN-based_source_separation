# Example of outputs
I trained Conv-TasNet and share the outputs. See `8sec_L64_H512/config/augmentation-*.yaml` for augmentation.
- You have to unzip `config.zip`, `log.zip`, and `loss.zip`.
- You can download pretrained models from [https://drive.google.com/file/d/1yQTdGu2jigAHotJJ7JAQcfqedACCP40t/view?usp=sharing](https://drive.google.com/file/d/1yQTdGu2jigAHotJJ7JAQcfqedACCP40t/view?usp=sharing) that includes `last.pth` and `best.pth`.
- You can download output JSON files from [https://drive.google.com/file/d/1R-63xiocr6Cjp84O9pKej-X2iZFb-QOl/view?usp=sharing](https://drive.google.com/file/d/1R-63xiocr6Cjp84O9pKej-X2iZFb-QOl/view?usp=sharing).

## Results
- SDR [dB] (median of median SDR of each song computed by `museval`)

| Model | Sampling rate [Hz] | Duration [sec] | L | N | H | Vocals | Drums | Bass | Other | Accompaniment | Average | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Conv-TasNet | 44100 | 8 | 64 | 256 | 512 | 5.91 | 5.82 | 5.38 | 3.51 | 11.85 | 5.16 | Epoch is chosen by validation loss. |